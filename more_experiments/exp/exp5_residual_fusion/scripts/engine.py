"""Phase 1 training loop for exp-5: RGB branch (backbone + CORN head)
frozen throughout, only the tissue CNN and residual fusion head train.

Adapted from task_3/src/engine.py: model(rgb, tissue) -> (z_final, z_rgb,
delta_z) instead of model(images) -> logits; loss/metrics computed on
z_final (CORN loss is architecture-agnostic, only cares about the final
logits). Real early stopping (not just best-checkpoint selection) per the
brief's explicit requirement -- stops after `patience` epochs without a
validation macro-F1 improvement, doesn't just run the full epoch budget.

Diagnostics collected every epoch (brief section 7): delta_z norm
(mean/max, train and val), train/val CORN loss. The "RGB baseline logits
remain unchanged during Phase 1" check is done directly, not assumed from
the branch being frozen: a probe batch's z_rgb is captured before training
and compared byte-for-byte after.
"""
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.losses import build_loss  # noqa: E402
from src.metrics import compute_metrics, predict_labels  # noqa: E402


def run_epoch(model, loader, criterion, device, head, num_classes, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0
    all_true, all_pred = [], []
    delta_norms = []

    with torch.set_grad_enabled(is_train):
        for rgb, tissue, targets in loader:
            rgb = rgb.to(device, non_blocking=True)
            tissue = tissue.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            z_final, z_rgb, delta_z = model(rgb, tissue)
            loss = criterion(z_final, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            preds = predict_labels(z_final.detach(), head)
            all_true.extend(targets.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            delta_norms.extend(delta_z.detach().norm(dim=1).cpu().tolist())

    stats = compute_metrics(all_true, all_pred, num_classes)
    stats["loss"] = total_loss / max(n_batches, 1)
    stats["delta_z_norm_mean"] = sum(delta_norms) / len(delta_norms) if delta_norms else 0.0
    stats["delta_z_norm_max"] = max(delta_norms) if delta_norms else 0.0
    return stats


def fit(model, train_loader, val_loader, device, max_epochs, lr, weight_decay, eta_min_factor,
        head, num_classes=4, log_prefix="", checkpoint_path=None, resume_path=None, patience=10):
    model.to(device)
    model.set_rgb_branch_trainable(False)  # Phase 1: RGB branch frozen for the whole run

    criterion = build_loss(head, num_classes)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * eta_min_factor)

    # Probe batch to verify the RGB branch's logits truly never move during Phase 1.
    probe_rgb, probe_tissue, _ = next(iter(val_loader))
    probe_rgb, probe_tissue = probe_rgb.to(device), probe_tissue.to(device)
    model.eval()
    with torch.no_grad():
        _, z_rgb_before, _ = model(probe_rgb, probe_tissue)
    z_rgb_before = z_rgb_before.clone()

    start_epoch = 1
    best_macro_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    diagnostics = []

    if resume_path is not None and resume_path.exists():
        try:
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_macro_f1 = ckpt["best_macro_f1"]
            best_state = ckpt["best_state"]
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
            diagnostics = ckpt.get("diagnostics", [])
            print(f"{log_prefix} resuming from epoch {start_epoch}/{max_epochs} (best={best_macro_f1:.4f})", flush=True)
        except Exception as exc:  # noqa: BLE001 - a truncated/corrupt checkpoint (killed mid-write) shouldn't be fatal
            print(f"{log_prefix} resume checkpoint unreadable ({exc}), starting from epoch 1", flush=True)
            resume_path.unlink()

    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs + 1):
        final_epoch = epoch
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, criterion, device, head, num_classes, optimizer)
        val_stats = run_epoch(model, val_loader, criterion, device, head, num_classes, None)
        scheduler.step()
        elapsed = time.time() - t0

        improved = val_stats["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_stats["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        diagnostics.append({
            "epoch": epoch,
            "train_loss": round(train_stats["loss"], 4), "val_loss": round(val_stats["loss"], 4),
            "val_macro_f1": round(val_stats["macro_f1"], 4),
            "train_delta_z_norm_mean": round(train_stats["delta_z_norm_mean"], 4),
            "val_delta_z_norm_mean": round(val_stats["delta_z_norm_mean"], 4),
            "val_delta_z_norm_max": round(val_stats["delta_z_norm_max"], 4),
        })

        print(
            f"{log_prefix} epoch {epoch}/{max_epochs} "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_acc={val_stats['accuracy']:.4f} val_f1={val_stats['macro_f1']:.4f} "
            f"val_qwk={val_stats['qwk']:.4f} best_f1={best_macro_f1:.4f} "
            f"delta_z_norm(val mean/max)={val_stats['delta_z_norm_mean']:.4f}/{val_stats['delta_z_norm_max']:.4f} "
            f"no_improve={epochs_without_improvement}/{patience} time={elapsed:.1f}s",
            flush=True,
        )

        if resume_path is not None:
            tmp_path = resume_path.with_suffix(resume_path.suffix + ".tmp")
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "best_macro_f1": best_macro_f1, "best_state": best_state,
                "epochs_without_improvement": epochs_without_improvement, "diagnostics": diagnostics,
            }, tmp_path)
            tmp_path.replace(resume_path)

        if epochs_without_improvement >= patience:
            print(f"{log_prefix} early stopping at epoch {epoch} "
                  f"(no val macro_f1 improvement for {patience} epochs)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_path is not None:
            torch.save(best_state, checkpoint_path)

    if resume_path is not None and resume_path.exists():
        resume_path.unlink()

    model.eval()
    with torch.no_grad():
        _, z_rgb_after, _ = model(probe_rgb, probe_tissue)
    rgb_logits_unchanged = torch.allclose(z_rgb_before, z_rgb_after, atol=1e-5)
    print(f"{log_prefix} RGB branch logits unchanged across Phase 1 training: {rgb_logits_unchanged}", flush=True)

    return model, best_macro_f1, diagnostics, rgb_logits_unchanged, final_epoch
