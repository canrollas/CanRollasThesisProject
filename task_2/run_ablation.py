#!/usr/bin/env python3
"""Wound stage classification ablation runner.

Cross-product of backbone encoder x classification head (plain softmax vs
CORN ordinal regression) from configs/resolved_grid.json, under 3-fold
stratified cross-validation on this project's wound-stage-classification
dataset. For every (config, fold) pair not already present in the results
CSV, trains on that fold's train/val split, evaluates on the held-out test
fold, and appends one row to the results CSV. Safe to interrupt and re-run
(already-completed rows are skipped), so the grid can be worked through
across multiple Colab sessions.
"""
import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR))

from src.augmentations import ImageAugmentation  # noqa: E402
from src.dataset import StageDataset  # noqa: E402
from src.engine import fit, run_epoch  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.model import build_model, count_params_m  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import (  # noqa: E402
    append_failure, append_result, get_device, load_completed_keys, load_yaml, set_seed, write_summary,
)


def make_loaders(data_root, fold, base_cfg, batch_size, workers, smoke):
    aug = ImageAugmentation(**base_cfg["augment"])
    train_samples, val_samples, test_samples = fold["train"], fold["val"], fold["test"]

    if smoke:
        train_samples, val_samples, test_samples = train_samples[:8], val_samples[:4], test_samples[:4]

    train_ds = StageDataset(train_samples, data_root, base_cfg["image_size"], transform=aug)
    val_ds = StageDataset(val_samples, data_root, base_cfg["image_size"], transform=None)
    test_ds = StageDataset(test_samples, data_root, base_cfg["image_size"], transform=None)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None, help="defaults to <repo>/datasets/wound-stage-classification")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--configs", default=None,
                    help="comma-separated config_id allowlist (from configs/resolved_grid.json), "
                         "e.g. 'resnet34__softmax,resnet34__corn' -- default: all")
    p.add_argument("--smoke", action="store_true", help="1 config, 1 fold, 2 epochs, tiny subset")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(TASK_DIR / "configs" / "base.yaml")
    resolved_path = TASK_DIR / "configs" / "resolved_grid.json"
    if not resolved_path.exists():
        raise SystemExit("configs/resolved_grid.json not found — run build_grid.py first")
    grid = json.loads(resolved_path.read_text())

    if args.configs is not None:
        wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
        by_id = {c["config_id"]: c for c in grid}
        missing = [c for c in wanted if c not in by_id]
        if missing:
            raise SystemExit(f"--configs: unknown config_id(s) not in resolved_grid.json: {missing}")
        grid = [by_id[c] for c in wanted]
    if args.smoke:
        grid = grid[:1]

    repo_root = TASK_DIR.parent
    data_root = Path(args.data_root) if args.data_root else repo_root / "datasets" / "wound-stage-classification"

    set_seed(base_cfg["seed"])
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    if args.smoke:
        folds = folds[:1]

    device = get_device()
    results_csv = TASK_DIR / base_cfg["results_csv"]
    failures_csv = results_csv.with_name("ablation_failures.csv")
    checkpoint_dir = TASK_DIR / base_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(results_csv)

    epochs = args.epochs if args.epochs is not None else (2 if args.smoke else base_cfg["epochs"])
    batch_size = args.batch if args.batch is not None else base_cfg["batch_size"]
    num_classes = len(base_cfg["classes"])

    print(f"device={device} configs={len(grid)} folds={len(folds)} epochs={epochs}", flush=True)

    for cfg in grid:
        for fold_idx, fold in enumerate(folds):
            if (cfg["config_id"], str(fold_idx)) in completed:
                continue

            model = None
            try:
                train_loader, val_loader, test_loader = make_loaders(
                    data_root, fold, base_cfg, batch_size, args.workers, args.smoke
                )

                model = build_model(cfg["encoder"], cfg["weights"], cfg["head"], num_classes=num_classes,
                                    dropout=base_cfg.get("head_dropout", 0.2))
                params_m = count_params_m(model)
                checkpoint_path = checkpoint_dir / f"{cfg['config_id']}_fold{fold_idx}.pt"
                resume_path = checkpoint_dir / f"{cfg['config_id']}_fold{fold_idx}.resume.pt"

                t0 = time.time()
                model, _ = fit(
                    model, train_loader, val_loader, device,
                    epochs=epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
                    eta_min_factor=base_cfg["eta_min_factor"],
                    head=cfg["head"], num_classes=num_classes,
                    log_prefix=f"[{cfg['config_id']} fold{fold_idx}]",
                    checkpoint_path=checkpoint_path,
                    resume_path=resume_path,
                )
                train_time_s = time.time() - t0

                criterion = build_loss(cfg["head"], num_classes)
                test_stats = run_epoch(model, test_loader, criterion, device, cfg["head"], num_classes,
                                        optimizer=None)

                row = {
                    "config_id": cfg["config_id"],
                    "encoder": cfg["encoder"],
                    "head": cfg["head"],
                    "weights": cfg["weights"],
                    "fold": fold_idx,
                    "params_m": round(params_m, 3),
                    "accuracy": round(test_stats["accuracy"], 4),
                    "macro_f1": round(test_stats["macro_f1"], 4),
                    "mae": round(test_stats["mae"], 4),
                    "qwk": round(test_stats["qwk"], 4),
                    "train_time_s": round(train_time_s, 1),
                }
                append_result(results_csv, row)
                write_summary(results_csv, results_csv.with_name("ablation_summary.csv"))
                print(f"[{cfg['config_id']} fold{fold_idx}] DONE test_acc={row['accuracy']:.4f} "
                      f"test_f1={row['macro_f1']:.4f} test_qwk={row['qwk']:.4f}", flush=True)

            except Exception as exc:  # noqa: BLE001 - one bad config must not take the whole grid down
                traceback.print_exc()
                print(f"[{cfg['config_id']} fold{fold_idx}] FAILED: {exc}", flush=True)
                append_failure(failures_csv, {
                    "config_id": cfg["config_id"],
                    "encoder": cfg["encoder"],
                    "head": cfg["head"],
                    "weights": cfg["weights"],
                    "fold": fold_idx,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })

            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
