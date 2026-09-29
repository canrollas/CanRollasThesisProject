import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# `!python script.py` in Colab pipes stdout, which switches Python to full
# block buffering instead of line buffering -- output then appears in
# delayed bursts instead of per print() call. Force line buffering so
# per-epoch progress (including engine.py's prints, same process/stdout)
# shows up immediately.
sys.stdout.reconfigure(line_buffering=True)

from src.augmentations import build_transform
from src.dataset import WoundRegionDataset, list_filenames, make_kfold_splits, split_holdout, subsample
from src.engine import run_training
from src.losses import build_loss
from src.model import build_model
from src.utils import apply_overrides, load_color_mapping, load_yaml, set_seed

STUDIES = ["backbone", "decoder", "weights", "loss", "augmentation", "combined"]


def build_loaders(config, train_files, val_files, test_files, device="cpu"):
    classes = config["data"]["classes"]
    color_mapping = load_color_mapping(config["data"]["color_mapping_path"], classes)
    resolution = config["train"]["resolution"]

    train_tf = build_transform(config["train"]["augmentation"], resolution, is_train=True)
    eval_tf = build_transform(config["train"]["augmentation"], resolution, is_train=False)

    common = dict(
        images_dir=config["data"]["images_dir"],
        masks_dir=config["data"]["masks_dir"],
        classes=classes,
        color_mapping=color_mapping,
    )
    train_ds = WoundRegionDataset(filenames=train_files, transform=train_tf, **common)
    val_ds = WoundRegionDataset(filenames=val_files, transform=eval_tf, **common)
    test_ds = WoundRegionDataset(filenames=test_files, transform=eval_tf, **common)

    bs = config["train"]["batch_size"]
    nw = config["train"].get("num_workers", 2)
    loader_kwargs = dict(
        num_workers=nw,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=nw > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, len(classes)


def run_one_fold(config, train_files, val_files, test_files, device, label=""):
    set_seed(config["train"]["seed"])
    train_loader, val_loader, test_loader, num_classes = build_loaders(
        config, train_files, val_files, test_files, device=device
    )

    model = build_model(config["model"]["decoder"], config["model"]["encoder"], config["model"]["weights"], num_classes)
    model.to(device)

    loss_fn = build_loss(config["train"]["loss"], config["data"]["ignore_index"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"]
    )
    epochs = config["train"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=config["train"]["lr"] * config["train"]["lr_min_factor"]
    )

    start = time.time()
    model, metrics = run_training(
        model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn,
        epochs, device, num_classes, config["data"]["ignore_index"], label=label,
        class_names=config["data"]["classes"],
    )
    metrics["train_time_sec"] = time.time() - start
    metrics["num_params"] = sum(p.numel() for p in model.parameters())
    return model, metrics


def mean_std(values):
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def load_completed_folds(fold_csv):
    """Returns {(case_index, fold_index): row_dict} for folds already
    written by a previous (possibly interrupted) run, so re-running a
    study never retrains work that is already on disk."""
    if not fold_csv.exists():
        return {}
    completed = {}
    with open(fold_csv, newline="") as f:
        for row in csv.DictReader(f):
            completed[(int(row["case_index"]), int(row["fold_index"]))] = row
    return completed


def run_study(study, base_config, folds, n_folds, device):
    study_config = load_yaml(f"configs/ablations/{study}.yaml")

    fold_csv = Path(base_config["output"]["results_csv"]).with_name(f"{study}_folds.csv")
    summary_csv = Path(base_config["output"]["results_csv"]).with_name(f"{study}_summary.csv")
    fold_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(base_config["output"]["checkpoint_dir"]) / study
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = load_completed_folds(fold_csv)
    if completed:
        print(f"[{study}] resuming: {len(completed)} fold(s) already completed in {fold_csv}")

    fold_rows = list(completed.values())
    summary_rows = []

    for i, case in enumerate(study_config["cases"]):
        overrides = case["overrides"]
        config = apply_overrides(base_config, overrides)
        print(f"[{study}] case {i + 1}/{len(study_config['cases'])}: {overrides}")

        case_test_miou, case_test_mdice, case_val_miou, case_train_time = [], [], [], []
        num_params = None

        for fold_idx, (train_files, val_files, test_files) in enumerate(folds):
            key = (i, fold_idx)
            if key in completed:
                row = completed[key]
                print(f"  fold {fold_idx + 1}/{n_folds} already done, skipping "
                      f"(test_miou={float(row['test_miou']):.4f})")
                case_test_miou.append(float(row["test_miou"]))
                case_test_mdice.append(float(row["test_mdice"]))
                case_val_miou.append(float(row["best_val_miou"]))
                case_train_time.append(float(row["train_time_sec"]))
                num_params = int(row["num_params"])
                continue

            train_files_fold = subsample(train_files, config["train"]["train_fraction"], config["train"]["seed"])
            print(f"  fold {fold_idx + 1}/{n_folds} "
                  f"(train={len(train_files_fold)}, val={len(val_files)}, test={len(test_files)})")

            label = f"{study} case{i}/fold{fold_idx} {json.dumps(overrides)}"
            model, metrics = run_one_fold(config, train_files_fold, val_files, test_files, device, label=label)
            num_params = metrics["num_params"]

            ckpt_path = ""
            if base_config["output"]["save_checkpoints"]:
                ckpt_path = checkpoint_dir / f"case_{i:02d}_fold{fold_idx}.pt"
                torch.save(model.state_dict(), ckpt_path)
                fold_config = dict(config)
                fold_config["_fold_index"] = fold_idx
                with open(checkpoint_dir / f"case_{i:02d}_fold{fold_idx}_config.json", "w") as f:
                    json.dump(fold_config, f, indent=2)

            fold_row = {
                "study": study,
                "case_index": i,
                "fold_index": fold_idx,
                "overrides": json.dumps(overrides),
                "test_miou": metrics["test"]["miou"],
                "test_mdice": metrics["test"]["mdice"],
                "test_iou_per_class": json.dumps(metrics["test"]["iou_per_class"]),
                "best_val_miou": metrics["best_val_miou"],
                "train_time_sec": metrics["train_time_sec"],
                "num_params": metrics["num_params"],
                "checkpoint": str(ckpt_path),
            }
            fold_rows.append(fold_row)
            case_test_miou.append(metrics["test"]["miou"])
            case_test_mdice.append(metrics["test"]["mdice"])
            case_val_miou.append(metrics["best_val_miou"])
            case_train_time.append(metrics["train_time_sec"])

            with open(fold_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(fold_row.keys()))
                writer.writeheader()
                writer.writerows(fold_rows)

        miou_mean, miou_std = mean_std(case_test_miou)
        mdice_mean, mdice_std = mean_std(case_test_mdice)
        val_miou_mean, val_miou_std = mean_std(case_val_miou)

        summary_row = {
            "study": study,
            "case_index": i,
            "overrides": json.dumps(overrides),
            "n_folds": n_folds,
            "test_miou_mean": miou_mean,
            "test_miou_std": miou_std,
            "test_mdice_mean": mdice_mean,
            "test_mdice_std": mdice_std,
            "best_val_miou_mean": val_miou_mean,
            "best_val_miou_std": val_miou_std,
            "train_time_sec_total": sum(case_train_time),
            "num_params": num_params,
        }
        summary_rows.append(summary_row)
        print(f"  -> test_miou={miou_mean:.4f}+-{miou_std:.4f}  test_mdice={mdice_mean:.4f}+-{mdice_std:.4f}")

        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"[{study}] done. Per-fold: {fold_csv}  Summary (mean+-std over {n_folds} folds): {summary_csv}")


def main():
    parser = argparse.ArgumentParser(description="Wound region segmentation ablation runner (k-fold CV)")
    parser.add_argument("--study", required=True, choices=STUDIES + ["all"],
                         help="Which axis to run, or 'all' to run every axis in sequence "
                              "(backbone, decoder, weights, loss, augmentation, combined).")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=None,
                         help="Override train.batch_size from the base config (e.g. --batch 64).")
    parser.add_argument("--workers", type=int, default=None,
                         help="Override train.num_workers (DataLoader worker processes) from the base config.")
    args = parser.parse_args()

    base_config = load_yaml(args.base_config)
    if args.batch is not None:
        base_config["train"]["batch_size"] = args.batch
        print(f"Overriding batch_size -> {args.batch}")
    if args.workers is not None:
        base_config["train"]["num_workers"] = args.workers
        print(f"Overriding num_workers -> {args.workers}")

    n_folds = base_config["data"]["n_folds"]
    assert n_folds >= 3, "n_folds must be at least 3 (paper protocol: 3, ideally 5)"

    all_filenames = list_filenames(base_config["data"]["images_dir"])
    dev_pool, holdout_files = split_holdout(
        all_filenames, base_config["data"]["holdout_fraction"], base_config["data"]["holdout_seed"]
    )
    print(f"Holdout: {len(holdout_files)} files set aside (never used by ablation), "
          f"dev pool: {len(dev_pool)} files")

    holdout_record = Path(base_config["output"]["results_csv"]).parent / "holdout_files.json"
    holdout_record.parent.mkdir(parents=True, exist_ok=True)
    if not holdout_record.exists():
        with open(holdout_record, "w") as f:
            json.dump({
                "holdout_fraction": base_config["data"]["holdout_fraction"],
                "holdout_seed": base_config["data"]["holdout_seed"],
                "files": holdout_files,
            }, f, indent=2)

    folds = make_kfold_splits(
        dev_pool, n_folds, base_config["data"]["split_seed"], base_config["data"]["val_fraction_of_remaining"]
    )

    studies_to_run = STUDIES if args.study == "all" else [args.study]
    for study in studies_to_run:
        run_study(study, base_config, folds, n_folds, args.device)

    if args.study == "all":
        print(f"\nAll {len(STUDIES)} studies done: {', '.join(STUDIES)}")


if __name__ == "__main__":
    main()
