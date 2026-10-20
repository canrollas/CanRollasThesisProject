"""Training loop for exp-4's two-input (RGB, tissue) late-fusion model.

Adapted from task_3/src/engine.py: same CORN loss / optimizer / scheduler
/ checkpoint-selection-on-val-macro_f1 / resume logic, just calling
model(rgb, tissue) instead of model(images), since the RGB and tissue
streams are no longer pre-concatenated into one tensor (that's exp-3's
job, not this one). build_loss/predict_labels/compute_metrics are
task_3/src's own, imported unchanged -- those only look at logits/
targets, architecture-agnostic.

Optional staged freeze -> fine-tune (off by default): for the first
`freeze_epochs` epochs, the RGB backbone's weights are frozen
(requires_grad=False) and only the tissue CNN + fusion head train; after
that, the backbone unfreezes and trains at `finetune_lr` (a separate,
typically lower, param-group LR) while the rest keeps the normal LR. Off
(freeze_epochs=0) runs the same single-phase, same-LR-for-everything
protocol as every other experiment in this project, for a clean first
comparison against the baseline and exp-3 -- staged freezing is an
explicit opt-in follow-up, not silently baked into the default run.
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

    with torch.set_grad_enabled(is_train):
        for rgb, tissue, targets in loader:
            rgb = rgb.to(device, non_blocking=True)
            tissue = tissue.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(rgb, tissue)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            preds = predict_labels(logits.detach(), head)
            all_true.extend(targets.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    stats = compute_metrics(all_true, all_pred, num_classes)
    stats["loss"] = total_loss / max(n_batches, 1)
    return stats


def fit(model, train_loader, val_loader, device, epochs, lr, weight_decay, eta_min_factor,
        head, num_classes=4, log_prefix="", checkpoint_path=None, resume_path=None,
        freeze_epochs=0, finetune_lr=None):
    model.to(device)
    criterion = build_loss(head, num_classes)

    backbone_params = list(model.rgb_backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("rgb_backbone.")]
    backbone_lr = finetune_lr if (freeze_epochs > 0 and finetune_lr is not None) else lr
    optimizer = AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": other_params, "lr": lr},
    ], weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * eta_min_factor)

    if freeze_epochs > 0:
        model.set_rgb_backbone_trainable(False)
        print(f"{log_prefix} RGB backbone frozen for the first {freeze_epochs} epoch(s)", flush=True)

    start_epoch = 1
    best_macro_f1 = -1.0
    best_state = None

    if resume_path is not None and resume_path.exists():
        try:
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_macro_f1 = ckpt["best_macro_f1"]
            best_state = ckpt["best_state"]
            if start_epoch > freeze_epochs:
                model.set_rgb_backbone_trainable(True)
            print(f"{log_prefix} resuming from epoch {start_epoch}/{epochs} (best={best_macro_f1:.4f})", flush=True)
        except Exception as exc:  # noqa: BLE001 - a truncated/corrupt checkpoint (killed mid-write) shouldn't be fatal
            print(f"{log_prefix} resume checkpoint unreadable ({exc}), starting from epoch 1", flush=True)
            resume_path.unlink()

    for epoch in range(start_epoch, epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            model.set_rgb_backbone_trainable(True)
            print(f"{log_prefix} RGB backbone unfrozen at epoch {epoch} (lr={backbone_lr})", flush=True)

        t0 = time.time()
        train_stats = run_epoch(model, train_loader, criterion, device, head, num_classes, optimizer)
        val_stats = run_epoch(model, val_loader, criterion, device, head, num_classes, None)
        scheduler.step()
        elapsed = time.time() - t0

        if val_stats["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_stats["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"{log_prefix} epoch {epoch}/{epochs} "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_acc={val_stats['accuracy']:.4f} val_f1={val_stats['macro_f1']:.4f} "
            f"val_qwk={val_stats['qwk']:.4f} best_f1={best_macro_f1:.4f} time={elapsed:.1f}s",
            flush=True,
        )

        if resume_path is not None:
            tmp_path = resume_path.with_suffix(resume_path.suffix + ".tmp")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_macro_f1": best_macro_f1,
                "best_state": best_state,
            }, tmp_path)
            tmp_path.replace(resume_path)

    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_path is not None:
            torch.save(best_state, checkpoint_path)

    if resume_path is not None and resume_path.exists():
        resume_path.unlink()

    return model, best_macro_f1
