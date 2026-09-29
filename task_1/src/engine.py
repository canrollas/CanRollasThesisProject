import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .losses import CombinedLoss
from .metrics import compute_confusion, iou_dice_from_confusion


def run_epoch(model, loader, criterion, device, optimizer=None, num_classes=3):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

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
            conf += compute_confusion(logits.detach(), targets, num_classes)

    iou, dice = iou_dice_from_confusion(conf.cpu().numpy())
    return {
        "loss": total_loss / max(n_batches, 1),
        "miou": float(iou.mean()),
        "mdice": float(dice.mean()),
        "iou_per_class": iou.tolist(),
        "dice_per_class": dice.tolist(),
    }


def fit(model, train_loader, val_loader, device, epochs, lr, weight_decay, eta_min_factor,
        ce_weight=0.5, dice_weight=0.5, num_classes=3, log_prefix="", checkpoint_path=None):
    model.to(device)
    criterion = CombinedLoss(num_classes=num_classes, ce_weight=ce_weight, dice_weight=dice_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * eta_min_factor)

    best_miou = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer, num_classes)
        val_stats = run_epoch(model, val_loader, criterion, device, None, num_classes)
        scheduler.step()
        elapsed = time.time() - t0

        if val_stats["miou"] > best_miou:
            best_miou = val_stats["miou"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"{log_prefix} epoch {epoch}/{epochs} "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_miou={val_stats['miou']:.4f} best={best_miou:.4f} time={elapsed:.1f}s",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_path is not None:
            torch.save(best_state, checkpoint_path)

    return model, best_miou
