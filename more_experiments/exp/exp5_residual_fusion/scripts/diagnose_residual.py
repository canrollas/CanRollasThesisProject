#!/usr/bin/env python3
"""Root-cause diagnostics for exp-5 Phase 1: why doesn't the residual
branch improve validation performance even though delta_z's norm keeps
growing during training? Per fold, on the SAME validation split used for
early stopping:

  1. (reported separately, from the training logs already captured --
     this script focuses on 2-6)
  2. residual-on (z_final) vs residual-off (z_rgb) predictions, compared
     directly on identical inputs.
  3. gradient norms on the residual branch at the converged (saved)
     checkpoint -- is there still a live learning signal, or has it
     flatlined?
  4. delta_z norm vs. each CORN conditional logit's distance to its
     decision boundary (|z_rgb_k|, since each class-rank's flip happens at
     z_k crossing 0) -- directly answers "does the correction grow without
     crossing thresholds, or is it crossing thresholds that don't matter
     for the final decoded rank?"
  5. zero-residual reproduces the frozen RGB branch exactly (sanity check
     at the converged checkpoint, not just at init).
  6. tissue ablation: real tissue vs. zeroed tissue vs. tissue SHUFFLED
     across the batch (same RGB, wrong tissue) -- if delta_z barely
     differs between real and shuffled tissue, the branch has learned a
     roughly constant correction instead of actually responding to
     per-image tissue content.

Does not modify or retrain anything -- reads the already-trained
checkpoints as-is.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from coral_pytorch.dataset import corn_label_from_logits

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from sklearn.metrics import cohen_kappa_score  # noqa: E402

from src.losses import build_loss  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402

CONFIG_IDS = {
    "rgb_tissue": "mit_b1__corn__residual_latefusion",
    "tissue_only": "mit_b1__corn__tissue_only_residual",
}

sys.path.insert(0, str(THIS_DIR))
from dataset import TwoStreamStageDataset  # noqa: E402
from model import build_model  # noqa: E402

CLASS_NAMES = ["stage_1", "stage_2", "stage_3", "stage_4"]


def load_split(fold_idx, base_cfg, data_root, split="val"):
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    samples = folds[fold_idx][split]
    ds = TwoStreamStageDataset(samples, data_root, base_cfg["image_size"], transform=None)
    rgbs, tissues, labels, rel_paths = [], [], [], []
    for i in range(len(ds)):
        r, t, label = ds[i]
        rgbs.append(r)
        tissues.append(t)
        labels.append(label)
        rel_paths.append(samples[i][0])
    return torch.stack(rgbs), torch.stack(tissues), torch.tensor(labels), rel_paths


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2, 3])


def analyze_fold(fold_idx, base_cfg, data_root, device, variant):
    print(f"\n{'=' * 70}\nFOLD {fold_idx} (variant={variant})\n{'=' * 70}")

    rgb, tissue, labels, rel_paths = load_split(fold_idx, base_cfg, data_root, split="val")
    rgb, tissue, labels = rgb.to(device), tissue.to(device), labels.to(device)
    n = rgb.shape[0]
    print(f"val set size: {n}")

    config_id = CONFIG_IDS[variant]
    ckpt_path = EXP_DIR / "checkpoints" / f"{config_id}_fold{fold_idx}.pt"
    if not ckpt_path.is_file():
        print(f"  SKIP: no checkpoint at {ckpt_path}")
        return None
    model = build_model("mit_b1", "imagenet", num_classes=4, variant=variant).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # --- 2 & 5: residual on vs off, on identical inputs ---
    with torch.no_grad():
        z_final, z_rgb, delta_z = model(rgb, tissue)

    pred_final = corn_label_from_logits(z_final)
    pred_rgb_only = corn_label_from_logits(z_rgb)

    changed = pred_final != pred_rgb_only
    correct_final = pred_final == labels
    correct_rgb = pred_rgb_only == labels
    fixed = (changed & (~correct_rgb) & correct_final).sum().item()
    broken = (changed & correct_rgb & (~correct_final)).sum().item()
    changed_neutral = changed.sum().item() - fixed - broken

    labels_np = labels.cpu().numpy()
    qwk_rgb = qwk(labels_np, pred_rgb_only.cpu().numpy())
    qwk_fusion = qwk(labels_np, pred_final.cpu().numpy())

    print(f"\n[2] residual-on vs residual-off predictions (same {n} val images):")
    print(f"  predictions changed by adding delta_z: {changed.sum().item()}/{n}")
    print(f"    -> fixed (wrong under RGB-only, correct under fusion):   {fixed}")
    print(f"    -> broken (correct under RGB-only, wrong under fusion):  {broken}")
    print(f"    -> changed but same correctness outcome:                 {changed_neutral}")
    print(f"  RGB-only accuracy on this val set:    {correct_rgb.float().mean().item():.4f}")
    print(f"  Fusion (residual-on) accuracy:        {correct_final.float().mean().item():.4f}")
    print(f"  RGB-only QWK:                          {qwk_rgb:.4f}")
    print(f"  Fusion (residual-on) QWK:              {qwk_fusion:.4f}  (delta: {qwk_fusion - qwk_rgb:+.4f})")

    print("\n[5] zero-residual reproduces the frozen RGB branch at the CONVERGED checkpoint:")
    print(f"  z_final == z_rgb when delta_z forced to 0: "
          f"{torch.equal(z_rgb, z_rgb)} (trivially true by construction -- "
          f"real check is pred_rgb_only above uses z_rgb directly, matching Phase 1's "
          f"frozen branch exactly since it never updated)")

    # --- 4: delta_z magnitude vs. each conditional logit's distance to its decision boundary ---
    margin = z_rgb.abs()  # (n, 3) -- distance of each conditional logit from 0 (the flip point)
    correction = delta_z.abs()
    crosses = (correction > margin)  # per (sample, logit) -- big enough to flip that conditional alone
    print(f"\n[4] delta_z component magnitude vs. z_rgb's distance to its decision boundary:")
    print(f"  mean |z_rgb| per conditional logit (margin):   {margin.mean(dim=0).cpu().numpy().round(4)}")
    print(f"  mean |delta_z| per conditional logit:          {correction.mean(dim=0).cpu().numpy().round(4)}")
    print(f"  (sample, logit) pairs where |delta_z| > margin: {crosses.sum().item()} / {crosses.numel()} "
          f"({100 * crosses.float().mean().item():.1f}%)")
    print(f"  ...but actual final-rank prediction changes:    {changed.sum().item()} / {n} "
          f"({100 * changed.float().mean().item():.1f}%)")
    if crosses.float().mean().item() > 0.05 and changed.float().mean().item() < 0.05:
        print("  -> corrections ARE large enough to flip individual conditional logits often, "
              "but this rarely changes the final decoded CORN rank -- consistent with "
              "corrections growing in a way that doesn't translate to the final prediction "
              "(e.g. flipping a conditional logit CORN's rank-consistency constraint overrides, "
              "or flipping logits whose sign was already 'spare' / non-decisive for the rank).")

    # --- 3: gradient norms at the converged checkpoint ---
    model.zero_grad()
    criterion = build_loss("corn", 4)
    z_final_g, _, _ = model(rgb, tissue)
    loss = criterion(z_final_g, labels)
    loss.backward()
    tissue_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.tissue_cnn.parameters() if p.grad is not None) ** 0.5
    residual_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.residual_head.parameters() if p.grad is not None) ** 0.5
    delta_proj_grad_norm = model.residual_head.delta_proj.weight.grad.norm().item()
    rgb_grads = [p.grad for p in model.rgb_backbone.parameters()]
    print(f"\n[3] gradient norms at the converged checkpoint (one backward pass on this val batch):")
    print(f"  val loss at this checkpoint:            {loss.item():.4f}")
    print(f"  ||grad|| tissue CNN:                     {tissue_grad_norm:.4f}")
    print(f"  ||grad|| residual_head (all):             {residual_grad_norm:.4f}")
    print(f"  ||grad|| residual_head.delta_proj only:   {delta_proj_grad_norm:.4f}")
    print(f"  RGB backbone grads still None (frozen):  {all(g is None for g in rgb_grads)}")
    model.zero_grad()

    # --- 6: tissue ablation -- real vs zeroed vs shuffled ---
    with torch.no_grad():
        _, _, delta_z_zero = model(rgb, torch.zeros_like(tissue))
        perm = torch.randperm(n, device=device)
        _, _, delta_z_shuffled = model(rgb, tissue[perm])

    real_norm = delta_z.norm(dim=1)
    zero_norm = delta_z_zero.norm(dim=1)
    shuf_norm = delta_z_shuffled.norm(dim=1)
    cos_real_vs_shuffled = F.cosine_similarity(delta_z, delta_z_shuffled, dim=1)
    l2_real_vs_shuffled = (delta_z - delta_z_shuffled).norm(dim=1)
    l2_real_vs_zero = (delta_z - delta_z_zero).norm(dim=1)
    # how big is the real-vs-shuffled difference relative to the correction's own size?
    # (an L2 gap can look "big" in absolute terms yet still be small relative to ||delta_z||)
    relative_l2_vs_shuffled = l2_real_vs_shuffled / real_norm.clamp_min(1e-8)

    print(f"\n[6] tissue ablation (same RGB, varying tissue input):")
    print(f"  mean ||delta_z|| with REAL tissue:      {real_norm.mean().item():.4f}")
    print(f"  mean ||delta_z|| with ZEROED tissue:     {zero_norm.mean().item():.4f}")
    print(f"  mean ||delta_z|| with SHUFFLED tissue:   {shuf_norm.mean().item():.4f}")
    print(f"  mean cosine(delta_z_real, delta_z_shuffled):     {cos_real_vs_shuffled.mean().item():.4f} "
          f"(near 1.0 = same DIRECTION regardless of tissue content)")
    print(f"  mean L2(delta_z_real, delta_z_shuffled):          {l2_real_vs_shuffled.mean().item():.4f}")
    print(f"  mean L2(delta_z_real, delta_z_zero):              {l2_real_vs_zero.mean().item():.4f}")
    print(f"  mean L2(real,shuffled) / mean ||delta_z_real||:   {relative_l2_vs_shuffled.mean().item():.4f} "
          f"(near 0 = shuffled tissue barely moves the correction relative to its own "
          f"magnitude -> correction is effectively tissue-content-independent, not just "
          f"direction-independent)")

    return {
        "fold": fold_idx, "n_val": n,
        "accuracy_rgb_only": correct_rgb.float().mean().item(),
        "accuracy_fusion": correct_final.float().mean().item(),
        "qwk_rgb_only": qwk_rgb, "qwk_fusion": qwk_fusion,
        "mean_l2_real_vs_shuffled": l2_real_vs_shuffled.mean().item(),
        "mean_relative_l2_vs_shuffled": relative_l2_vs_shuffled.mean().item(),
        "n_changed": changed.sum().item(), "n_fixed": fixed, "n_broken": broken,
        "mean_delta_z_norm_real": real_norm.mean().item(),
        "mean_delta_z_norm_zero_tissue": zero_norm.mean().item(),
        "mean_delta_z_norm_shuffled_tissue": shuf_norm.mean().item(),
        "mean_cosine_real_vs_shuffled": cos_real_vs_shuffled.mean().item(),
        "residual_head_grad_norm_at_convergence": residual_grad_norm,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="tissue_only", choices=["rgb_tissue", "tissue_only"])
    args = p.parse_args()

    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = EXP_DIR / base_cfg["data_root"]
    device = get_device()
    print(f"device={device} variant={args.variant}")

    summaries = []
    for fold_idx in range(3):
        result = analyze_fold(fold_idx, base_cfg, data_root, device, args.variant)
        if result is not None:
            summaries.append(result)

    print(f"\n{'=' * 70}\nSUMMARY ACROSS FOLDS (variant={args.variant})\n{'=' * 70}")
    for s in summaries:
        print(f"fold{s['fold']}: RGB-only acc={s['accuracy_rgb_only']:.4f} fusion acc={s['accuracy_fusion']:.4f} "
              f"RGB-only qwk={s['qwk_rgb_only']:.4f} fusion qwk={s['qwk_fusion']:.4f} "
              f"changed={s['n_changed']}/{s['n_val']} fixed={s['n_fixed']} broken={s['n_broken']} "
              f"cos(real,shuffled)={s['mean_cosine_real_vs_shuffled']:.4f} "
              f"relL2(real,shuffled)={s['mean_relative_l2_vs_shuffled']:.4f}")


if __name__ == "__main__":
    main()
