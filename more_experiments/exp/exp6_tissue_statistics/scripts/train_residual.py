#!/usr/bin/env python3
"""tissue_statistics_residual_fusion (Chapter 8, exp-6): frozen trained RGB
MiT-B1+CORN baseline + a small residual MLP on the 20-dim hand-crafted
tissue-statistics vector, zero-init final layer (z_final = z_rgb +
delta_z, reproduces the baseline exactly at construction -- verified
against real data before training, same as exp-5).

Reuses exp5/scripts/engine.py UNCHANGED: StatsResidualModel's
forward(rgb, stats) -> (z_final, z_rgb, delta_z) matches that engine's
expected signature exactly.
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

THIS_DIR = Path(__file__).resolve().parent  # exp/exp6_tissue_statistics/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.augmentations import ImageAugmentation  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.model import build_model as build_rgb_only_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml, set_seed  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from dataset import TissueStatsResidualDataset, load_feature_table  # noqa: E402
from model import build_residual_model, count_params_m, load_rgb_baseline  # noqa: E402

sys.path.insert(0, str((REPO_ROOT / "more_experiments" / "exp" / "exp5_residual_fusion" / "scripts")))
from engine import fit, run_epoch  # noqa: E402

CONFIG_ID = "mit_b1__corn__tissue_statistics_residual_fusion"
RGB_BASELINE_CONFIG_ID = "mit_b1__corn"


def make_loaders(data_root, fold, feature_lookup, base_cfg, batch_size, workers, smoke):
    aug = ImageAugmentation(**base_cfg["augment"])
    train_samples, val_samples, test_samples = fold["train"], fold["val"], fold["test"]
    if smoke:
        train_samples, val_samples, test_samples = train_samples[:8], val_samples[:4], test_samples[:4]

    train_ds = TissueStatsResidualDataset(train_samples, data_root, feature_lookup, base_cfg["image_size"], transform=aug)
    val_ds = TissueStatsResidualDataset(val_samples, data_root, feature_lookup, base_cfg["image_size"], transform=None)
    test_ds = TissueStatsResidualDataset(test_samples, data_root, feature_lookup, base_cfg["image_size"], transform=None)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def verify_reproduces_baseline(model, rgb_checkpoint_path, probe_rgb, probe_stats, device, num_classes=4):
    reference = build_rgb_only_model("mit_b1", None, "corn", num_classes=num_classes)
    reference.load_state_dict(torch.load(rgb_checkpoint_path, map_location=device))
    reference.to(device).eval()
    model.eval()
    with torch.no_grad():
        reference_logits = reference(probe_rgb)
        z_final, z_rgb, delta_z = model(probe_rgb, probe_stats)
    return {
        "fusion z_rgb matches reference exactly": torch.equal(z_rgb, reference_logits),
        "fusion z_final matches reference exactly (delta_z==0 at init)": torch.equal(z_final, reference_logits),
        "delta_z is exactly zero at init": torch.equal(delta_z, torch.zeros_like(delta_z)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--folds", default=None)
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
    checkpoint_dir = EXP_DIR / base_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = args.max_epochs if args.max_epochs is not None else (2 if args.smoke else base_cfg["max_epochs"])
    patience = args.patience if args.patience is not None else base_cfg["patience"]
    batch_size = args.batch if args.batch is not None else base_cfg["batch_size"]
    num_classes = len(base_cfg["classes"])
    wanted_folds = set(int(x) for x in args.folds.split(",")) if args.folds else None

    print(f"device={device} config={CONFIG_ID} folds={len(folds)} max_epochs={max_epochs} patience={patience}", flush=True)

    results_rows = []
    for fold_idx, fold in enumerate(folds):
        if wanted_folds is not None and fold_idx not in wanted_folds:
            continue

        rgb_checkpoint_path = (EXP_DIR / base_cfg["rgb_checkpoint_template"].format(fold=fold_idx)).resolve()
        if not rgb_checkpoint_path.is_file():
            print(f"[fold{fold_idx}] SKIP: RGB baseline checkpoint not found at {rgb_checkpoint_path} "
                  f"-- train with: cd task_3 && python run_ablation.py --configs {RGB_BASELINE_CONFIG_ID}",
                  file=sys.stderr)
            continue

        print(f"[fold{fold_idx}] loading RGB baseline from {rgb_checkpoint_path}", flush=True)
        train_loader, val_loader, test_loader = make_loaders(
            data_root, fold, feature_lookup, base_cfg, batch_size, args.workers, args.smoke
        )

        model = build_residual_model(base_cfg["encoder"], base_cfg["weights"], num_classes=num_classes)
        load_rgb_baseline(model, rgb_checkpoint_path, device)
        model.to(device)

        probe_rgb, probe_stats, _ = next(iter(val_loader))
        probe_rgb, probe_stats = probe_rgb.to(device), probe_stats.to(device)
        checks = verify_reproduces_baseline(model, rgb_checkpoint_path, probe_rgb, probe_stats, device, num_classes)
        for name, ok in checks.items():
            print(f"[fold{fold_idx}] [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        if not all(checks.values()):
            raise SystemExit(f"[fold{fold_idx}] STOP: residual-fusion model doesn't reproduce the baseline exactly.")

        params_m = count_params_m(model)
        checkpoint_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.pt"
        resume_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.resume.pt"

        t0 = time.time()
        model, best_val_f1, diagnostics, rgb_unchanged, final_epoch = fit(
            model, train_loader, val_loader, device,
            max_epochs=max_epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
            eta_min_factor=base_cfg["eta_min_factor"], head="corn", num_classes=num_classes,
            log_prefix=f"[{CONFIG_ID} fold{fold_idx}]",
            checkpoint_path=checkpoint_path, resume_path=resume_path, patience=patience,
        )
        train_time_s = time.time() - t0

        criterion = build_loss("corn", num_classes)
        test_stats = run_epoch(model, test_loader, criterion, device, "corn", num_classes, optimizer=None)

        with open(checkpoint_path.with_suffix(".provenance.json"), "w") as f:
            json.dump({
                "config_id": CONFIG_ID, "fold": fold_idx,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rgb_baseline_checkpoint": str(rgb_checkpoint_path.relative_to(REPO_ROOT)),
                "pretraining_verification": checks, "early_stopped_at_epoch": final_epoch,
                "rgb_logits_unchanged_across_training": rgb_unchanged, "params_m": round(params_m, 3),
            }, f, indent=2)

        row = {
            "config_id": CONFIG_ID, "encoder": base_cfg["encoder"], "head": "corn",
            "weights": "baseline_checkpoint", "fold": fold_idx, "params_m": round(params_m, 3),
            "accuracy": round(test_stats["accuracy"], 4), "macro_f1": round(test_stats["macro_f1"], 4),
            "mae": round(test_stats["mae"], 4), "qwk": round(test_stats["qwk"], 4),
            "early_stopped_at_epoch": final_epoch, "train_time_s": round(train_time_s, 1),
        }
        results_rows.append(row)
        print(f"[{CONFIG_ID} fold{fold_idx}] DONE test_acc={row['accuracy']:.4f} "
              f"test_f1={row['macro_f1']:.4f} test_qwk={row['qwk']:.4f} (stopped epoch {final_epoch})", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    import csv
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config_id", "encoder", "head", "weights", "fold", "params_m",
                  "accuracy", "macro_f1", "mae", "qwk", "early_stopped_at_epoch", "train_time_s"]
    existing = {}
    if results_csv.exists():
        with open(results_csv, newline="") as f:
            for row in csv.DictReader(f):
                existing[(row["config_id"], int(row["fold"]))] = row
    for row in results_rows:
        existing[(row["config_id"], row["fold"])] = row
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(existing):
            writer.writerow(existing[key])
    print(f"wrote {results_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
