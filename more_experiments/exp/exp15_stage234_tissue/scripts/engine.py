"""Task 3's optimizer/loss/selection protocol with two inputs and resumable RNG."""
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.losses import build_loss
from src.metrics import compute_metrics, predict_labels


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run_epoch(model, loader, device, optimizer=None, collect=False):
    model.train(optimizer is not None)
    criterion = build_loss("corn", 3)
    total_loss, batches = 0.0, 0
    true, pred, probabilities = [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for rgb, tissue, targets in loader:
            rgb, tissue, targets = (x.to(device, non_blocking=True) for x in (rgb, tissue, targets))
            logits = model(rgb, tissue)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Three-class CORN must produce exactly two logits")
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite CORN loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            batches += 1
            true.extend(targets.detach().cpu().tolist())
            pred.extend(predict_labels(logits.detach(), "corn").cpu().tolist())
            if collect:
                survival = logits.detach().sigmoid().cumprod(dim=1)
                probs = torch.stack([1 - survival[:, 0], survival[:, 0] - survival[:, 1], survival[:, 1]], dim=1)
                probabilities.extend(probs.cpu().tolist())
    if not true:
        raise ValueError("Cannot evaluate an empty loader")
    stats = compute_metrics(true, pred, num_classes=3)
    stats["loss"] = total_loss / batches
    if collect:
        stats.update(y_true=true, y_pred=pred, probabilities=probabilities)
    return stats


def capture_rng(train_generator):
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(), "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator": train_generator.get_state(),
    }


def restore_rng(state, train_generator):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    train_generator.set_state(state["train_generator"])


def fit(model, train_loader, val_loader, device, config, checkpoint_path, resume_path,
        protocol_fingerprint, history_path, log_prefix=""):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lr"] * config["eta_min_factor"])
    start_epoch, best_epoch, best_f1, history, elapsed_total = 1, 0, -1.0, [], 0.0
    if resume_path.exists():
        # Only primitive Python values and tensors are saved in this checkpoint.
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        if state["protocol_fingerprint"] != protocol_fingerprint:
            raise ValueError("Resume checkpoint belongs to a different experiment protocol")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = state["epoch"] + 1
        best_epoch, best_f1 = state["best_epoch"], state["best_macro_f1"]
        history, elapsed_total = state["history"], state["train_time_s"]
        restore_rng(state["rng"], train_loader.generator)
        print(f"{log_prefix} resuming epoch {start_epoch}/{config['epochs']}", flush=True)

    for epoch in range(start_epoch, config["epochs"] + 1):
        t0 = time.monotonic()
        current_lr = optimizer.param_groups[0]["lr"]
        train_stats = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_stats = run_epoch(model, val_loader, device)
        scheduler.step()
        elapsed_total += time.monotonic() - t0
        if val_stats["macro_f1"] > best_f1:
            best_epoch, best_f1 = epoch, val_stats["macro_f1"]
            atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint_path)
        history.append({
            "epoch": epoch, "lr": current_lr, "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"], "val_accuracy": val_stats["accuracy"],
            "val_macro_f1": val_stats["macro_f1"], "val_mae": val_stats["mae"], "val_qwk": val_stats["qwk"],
            "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        })
        print(f"{log_prefix} epoch {epoch}/{config['epochs']} train_loss={train_stats['loss']:.4f} "
              f"val_f1={val_stats['macro_f1']:.4f} val_acc={val_stats['accuracy']:.4f} "
              f"best_f1={best_f1:.4f} best_epoch={best_epoch}", flush=True)
        atomic_save({
            "protocol_fingerprint": protocol_fingerprint, "epoch": epoch,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "best_epoch": best_epoch,
            "best_macro_f1": best_f1, "history": history, "train_time_s": elapsed_total,
            "rng": capture_rng(train_loader.generator),
        }, resume_path)

    if not checkpoint_path.exists():
        raise RuntimeError("Best validation checkpoint was not saved")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    # Keep the resume file until the caller commits the test report successfully.
    return {"best_epoch": best_epoch, "best_val_macro_f1": best_f1, "train_time_s": elapsed_total}
