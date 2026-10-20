#!/usr/bin/env python3
"""tissue_statistics_only (Chapter 8, exp-6): stage prediction from the
20-dim hand-crafted tissue-statistics vector ALONE -- no RGB input
anywhere. Isolates whether the tissue probability maps carry any staging
signal independent of appearance, before trying to fuse them with RGB.

Reuses task_3/src/engine.py's fit/run_epoch UNCHANGED: a StatsOnlyModel's
forward(stats) -> logits has the exact same shape as task_3's own
model(images) -> logits, so nothing about the training loop needed to
change, just the input tensor's meaning.

config_id: mit_b1__corn__tissue_statistics_only (kept consistent naming
even though there's no MiT-B1/image encoder involved here -- this is
intentional: the "encoder" field is just descriptive of nothing, the
point of the name is comparability with the other config_ids in
more_experiments/results/comparison.csv).
"""
import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent  # exp/exp6_tissue_statistics/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.engine import fit, run_epoch  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import (  # noqa: E402
    append_failure, append_result, get_device, load_completed_keys, load_yaml, set_seed, write_summary,
)

sys.path.insert(0, str(THIS_DIR))
from dataset import TissueStatsOnlyDataset, load_feature_table  # noqa: E402
from model import build_stats_only_model, count_params_m  # noqa: E402

CONFIG_ID = "mit_b1__corn__tissue_statistics_only"


def make_loaders(fold, feature_lookup, batch_size, workers, smoke):
    train_samples, val_samples, test_samples = fold["train"], fold["val"], fold["test"]
    if smoke:
        train_samples, val_samples, test_samples = train_samples[:8], val_samples[:4], test_samples[:4]

    train_ds = TissueStatsOnlyDataset(train_samples, feature_lookup)
    val_ds = TissueStatsOnlyDataset(val_samples, feature_lookup)
    test_ds = TissueStatsOnlyDataset(test_samples, feature_lookup)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = EXP_DIR / base_cfg["data_root"]
    feature_lookup = load_feature_table(EXP_DIR / base_cfg["feature_csv"])

    set_seed(base_cfg["seed"])
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    if args.smoke:
        folds = folds[:1]

    device = get_device()
    results_csv = EXP_DIR / base_cfg["results_csv"]
    failures_csv = results_csv.with_name("exp6_stats_only_failures.csv")
    checkpoint_dir = EXP_DIR / base_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(results_csv)

    epochs = args.epochs if args.epochs is not None else (2 if args.smoke else base_cfg["max_epochs"])
    batch_size = args.batch if args.batch is not None else base_cfg["batch_size"]
    num_classes = len(base_cfg["classes"])

    print(f"device={device} config={CONFIG_ID} folds={len(folds)} epochs={epochs}", flush=True)

    for fold_idx, fold in enumerate(folds):
        if (CONFIG_ID, str(fold_idx)) in completed:
            print(f"[{CONFIG_ID} fold{fold_idx}] already completed, skipping", flush=True)
            continue

        model = None
        try:
            train_loader, val_loader, test_loader = make_loaders(
                fold, feature_lookup, batch_size, args.workers, args.smoke
            )

            model = build_stats_only_model(num_classes=num_classes)
            params_m = count_params_m(model)
            checkpoint_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.pt"
            resume_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.resume.pt"

            t0 = time.time()
            model, _ = fit(
                model, train_loader, val_loader, device,
                epochs=epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
                eta_min_factor=base_cfg["eta_min_factor"],
                head="corn", num_classes=num_classes,
                log_prefix=f"[{CONFIG_ID} fold{fold_idx}]",
                checkpoint_path=checkpoint_path, resume_path=resume_path,
            )
            train_time_s = time.time() - t0

            criterion = build_loss("corn", num_classes)
            test_stats = run_epoch(model, test_loader, criterion, device, "corn", num_classes, optimizer=None)

            row = {
                "config_id": CONFIG_ID, "encoder": "none (stats-only MLP)", "head": "corn", "weights": "n/a",
                "fold": fold_idx, "params_m": round(params_m, 3),
                "accuracy": round(test_stats["accuracy"], 4), "macro_f1": round(test_stats["macro_f1"], 4),
                "mae": round(test_stats["mae"], 4), "qwk": round(test_stats["qwk"], 4),
                "train_time_s": round(train_time_s, 1),
            }
            append_result(results_csv, row)
            write_summary(results_csv, results_csv.with_name("exp6_stats_only_summary.csv"))
            print(f"[{CONFIG_ID} fold{fold_idx}] DONE test_acc={row['accuracy']:.4f} "
                  f"test_f1={row['macro_f1']:.4f} test_qwk={row['qwk']:.4f}", flush=True)

        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(f"[{CONFIG_ID} fold{fold_idx}] FAILED: {exc}", flush=True)
            append_failure(failures_csv, {
                "config_id": CONFIG_ID, "encoder": "none", "head": "corn", "weights": "n/a",
                "fold": fold_idx, "error": f"{type(exc).__name__}: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
