import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .losses import build_loss
from .metrics import compute_metrics, predict_labels


def run_epoch(model, loader, criterion, device, head, num_classes, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0
    all_true, all_pred = [], []

    with torch.set_grad_enabled(is_train):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
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
        head, num_classes=4, log_prefix="", checkpoint_path=None, resume_path=None):
    model.to(device)
    criterion = build_loss(head, num_classes)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * eta_min_factor)

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
            print(f"{log_prefix} resuming from epoch {start_epoch}/{epochs} (best={best_macro_f1:.4f})", flush=True)
        except Exception as exc:  # noqa: BLE001 - a truncated/corrupt checkpoint (killed mid-write) shouldn't be fatal
            print(f"{log_prefix} resume checkpoint unreadable ({exc}), starting from epoch 1", flush=True)
            resume_path.unlink()

    for epoch in range(start_epoch, epochs + 1):
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
