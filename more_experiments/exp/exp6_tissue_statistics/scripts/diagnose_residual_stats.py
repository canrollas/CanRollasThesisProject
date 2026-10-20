#!/usr/bin/env python3
"""Tissue-shuffle ablation for exp-6's residual-fusion checkpoints (step 5
of the brief): real vs. SHUFFLED 20-dim tissue-statistics vectors (same
RGB, a different image's stats), reporting cosine similarity, normalized
L2 difference, and prediction-flip counts -- same method as exp-5's
diagnose_residual.py, adapted for the hand-crafted-feature model.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from coral_pytorch.dataset import corn_label_from_logits
from sklearn.metrics import cohen_kappa_score

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from dataset import TissueStatsResidualDataset, load_feature_table  # noqa: E402
from model import build_residual_model  # noqa: E402

CONFIG_ID = "mit_b1__corn__tissue_statistics_residual_fusion"


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2, 3])


def analyze_fold(fold_idx, base_cfg, data_root, feature_lookup, device):
    ckpt_path = EXP_DIR / "checkpoints" / f"{CONFIG_ID}_fold{fold_idx}.pt"
    if not ckpt_path.is_file():
        print(f"\nfold{fold_idx}: SKIP, no checkpoint at {ckpt_path}")
        return None

    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    samples = folds[fold_idx]["val"]
    ds = TissueStatsResidualDataset(samples, data_root, feature_lookup, base_cfg["image_size"], transform=None)
    rgbs, stats, labels = [], [], []
    for i in range(len(ds)):
        r, s, label = ds[i]
        rgbs.append(r)
        stats.append(s)
        labels.append(label)
    rgb = torch.stack(rgbs).to(device)
    stats = torch.stack(stats).to(device)
    labels = torch.tensor(labels).to(device)
    n = rgb.shape[0]

    model = build_residual_model("mit_b1", "imagenet", num_classes=4).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        z_final, z_rgb, delta_z = model(rgb, stats)
        perm = torch.randperm(n, device=device)
        _, _, delta_z_shuffled = model(rgb, stats[perm])

    pred_final = corn_label_from_logits(z_final)
    pred_rgb_only = corn_label_from_logits(z_rgb)
    changed = pred_final != pred_rgb_only
    correct_final = pred_final == labels
    correct_rgb = pred_rgb_only == labels
    fixed = (changed & (~correct_rgb) & correct_final).sum().item()
    broken = (changed & correct_rgb & (~correct_final)).sum().item()

    real_norm = delta_z.norm(dim=1)
    cos_sim = F.cosine_similarity(delta_z, delta_z_shuffled, dim=1)
    l2_diff = (delta_z - delta_z_shuffled).norm(dim=1)
    rel_l2 = l2_diff / real_norm.clamp_min(1e-8)

    labels_np = labels.cpu().numpy()
    print(f"\nfold{fold_idx} (n_val={n}):")
    print(f"  RGB-only acc={correct_rgb.float().mean().item():.4f} qwk={qwk(labels_np, pred_rgb_only.cpu().numpy()):.4f}")
    print(f"  fusion   acc={correct_final.float().mean().item():.4f} qwk={qwk(labels_np, pred_final.cpu().numpy()):.4f}")
    print(f"  predictions changed: {changed.sum().item()}/{n}  fixed={fixed}  broken={broken}")
    print(f"  mean ||delta_z|| (real): {real_norm.mean().item():.4f}")
    print(f"  mean cosine(real, shuffled):        {cos_sim.mean().item():.4f}")
    print(f"  mean L2(real, shuffled):             {l2_diff.mean().item():.4f}")
    print(f"  mean relative L2 (L2/||delta_z||):  {rel_l2.mean().item():.4f}")

    return {
        "fold": fold_idx, "n_changed": changed.sum().item(), "fixed": fixed, "broken": broken,
        "cosine_real_shuffled": cos_sim.mean().item(), "rel_l2_real_shuffled": rel_l2.mean().item(),
    }


def main():
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = EXP_DIR / base_cfg["data_root"]
    feature_lookup = load_feature_table(EXP_DIR / base_cfg["feature_csv"])
    device = get_device()
    print(f"device={device}")

    for fold_idx in range(3):
        analyze_fold(fold_idx, base_cfg, data_root, feature_lookup, device)


if __name__ == "__main__":
    main()
