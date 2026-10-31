#!/usr/bin/env python3
"""Work Package 1 (advisor extended experiments): exp-5 architecture
(residual fusion, "tissue_only" variant), repeated with all 5 tissue
channels collapsed into ONE combined channel (per-pixel max) instead of
concatenated separately.

Per the advisor's note ("Tek kanal exp-4 ve exp-5 denenecek" -- 5 kanal
yerine tum dokular tek kanalda verilsin), Phase 1 only, exactly like exp-5
itself: the RGB branch (backbone + CORN head) is loaded from task_3's own
trained per-fold baseline checkpoint and stays frozen for the whole run;
only the tissue CNN + residual head train, with real early stopping on
val macro-F1. Before training starts for each fold, verifies the freshly
-built fusion model reproduces the loaded baseline's logits exactly
(z_final == z_rgb, delta_z == 0 at construction) against REAL data --
same two-step verification exp-5's own train.py does.

Writes results.csv with one row per fold
(config_id=mit_b1__corn__wp1_exp5tissueonly__tissue=combined_max), merged
by fold key so re-running a completed fold is a no-op overwrite, not a
duplicate row -- same merge pattern exp-5's own train.py uses for its two
variants.
"""
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent  # experiment_extended/wp1_single_tissue_channel/exp5_tissue_only/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[4]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.augmentations import ImageAugmentation  # noqa: E402
from src.model import build_model as build_rgb_only_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml, set_seed  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from dataset import TwoStreamCombinedTissueDataset  # noqa: E402
from engine import fit, run_epoch  # noqa: E402
from model import build_model, count_params_m, load_rgb_baseline  # noqa: E402

import argparse  # noqa: E402

RGB_BASELINE_CONFIG_ID = "mit_b1__corn"
CONFIG_ID = "mit_b1__corn__wp1_exp5tissueonly__tissue=combined_max"

RESULT_FIELDNAMES = ["config_id", "tissue_channel", "encoder", "head", "weights", "fold", "params_m",
                      "accuracy", "macro_f1", "mae", "qwk", "early_stopped_at_epoch", "train_time_s"]


def make_loaders(data_root, fold, base_cfg, batch_size, workers, smoke):
    aug = ImageAugmentation(**base_cfg["augment"])
    train_samples, val_samples, test_samples = fold["train"], fold["val"], fold["test"]

    if smoke:
        train_samples, val_samples, test_samples = train_samples[:8], val_samples[:4], test_samples[:4]

    train_ds = TwoStreamCombinedTissueDataset(train_samples, data_root, base_cfg["image_size"], transform=aug)
    val_ds = TwoStreamCombinedTissueDataset(val_samples, data_root, base_cfg["image_size"], transform=None)
    test_ds = TwoStreamCombinedTissueDataset(test_samples, data_root, base_cfg["image_size"], transform=None)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def verify_reproduces_baseline(fusion_model, rgb_checkpoint_path, probe_rgb, device, num_classes=4):
    reference = build_rgb_only_model("mit_b1", None, "corn", num_classes=num_classes)
    state_dict = torch.load(rgb_checkpoint_path, map_location=device)
    reference.load_state_dict(state_dict, strict=True)
    reference.to(device).eval()

    fusion_model.eval()
    with torch.no_grad():
        reference_logits = reference(probe_rgb)
        dummy_tissue = torch.zeros(probe_rgb.shape[0], 1, *probe_rgb.shape[2:], device=device)
        z_final, z_rgb, delta_z = fusion_model(probe_rgb, dummy_tissue)

    return {
        "fusion z_rgb matches reference exactly": torch.equal(z_rgb, reference_logits),
        "fusion z_final matches reference exactly (delta_z==0 at init)": torch.equal(z_final, reference_logits),
        "delta_z is exactly zero at init": torch.equal(delta_z, torch.zeros_like(delta_z)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--folds", default=None, help="comma-separated fold indices, default: all")
    p.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs, tiny subset -- sanity check")
    return p.parse_args()


def write_merged_csv(results_csv, new_rows):
    existing_by_key = {}
    if results_csv.exists():
        with open(results_csv, newline="") as f:
            for row in csv.DictReader(f):
                existing_by_key[(row["config_id"], int(row["fold"]))] = row
    for row in new_rows:
        existing_by_key[(row["config_id"], row["fold"])] = row
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for key in sorted(existing_by_key, key=lambda k: (k[0], k[1])):
            writer.writerow(existing_by_key[key])


def main():
    args = parse_args()
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    device = get_device()
    results_csv = EXP_DIR / base_cfg["results_csv"]
    checkpoint_dir = EXP_DIR / base_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wanted_folds = set(int(x) for x in args.folds.split(",")) if args.folds else None

    data_root = Path(args.data_root) if args.data_root else EXP_DIR / base_cfg["data_root"]

    set_seed(base_cfg["seed"])
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    if args.smoke:
        folds = folds[:1]

    max_epochs = args.max_epochs if args.max_epochs is not None else (2 if args.smoke else base_cfg["max_epochs"])
    patience = args.patience if args.patience is not None else base_cfg["patience"]
    batch_size = args.batch if args.batch is not None else base_cfg["batch_size"]
    num_classes = len(base_cfg["classes"])

    print(f"device={device} config={CONFIG_ID} folds={len(folds)} max_epochs={max_epochs} patience={patience}",
          flush=True)

    new_rows = []
    for fold_idx, fold in enumerate(folds):
        if wanted_folds is not None and fold_idx not in wanted_folds:
            continue
        rgb_checkpoint_path = (EXP_DIR / base_cfg["rgb_checkpoint_template"].format(fold=fold_idx)).resolve()
        if not rgb_checkpoint_path.is_file():
            raise SystemExit(
                f"\nBLOCKED: RGB baseline checkpoint not found for fold {fold_idx}: {rgb_checkpoint_path}\n"
                f"This experiment requires task_3's own trained mit_b1__corn checkpoint per fold "
                f"(config_id={RGB_BASELINE_CONFIG_ID!r}) -- it is not retrained here."
            )

        print(f"[{CONFIG_ID} fold{fold_idx}] loading RGB baseline from {rgb_checkpoint_path}", flush=True)
        train_loader, val_loader, test_loader = make_loaders(
            data_root, fold, base_cfg, batch_size, args.workers, args.smoke
        )

        model = build_model(base_cfg["encoder"], base_cfg["weights"], num_classes=num_classes)
        load_rgb_baseline(model, rgb_checkpoint_path, device)
        model.to(device)

        probe_rgb, _, _ = next(iter(val_loader))
        probe_rgb = probe_rgb.to(device)
        checks = verify_reproduces_baseline(model, rgb_checkpoint_path, probe_rgb, device, num_classes)
        for name, ok in checks.items():
            print(f"[{CONFIG_ID} fold{fold_idx}] [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        if not all(checks.values()):
            raise SystemExit(f"[{CONFIG_ID} fold{fold_idx}] STOP: does not reproduce the RGB baseline exactly.")

        params_m = count_params_m(model)
        checkpoint_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.pt"
        resume_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.resume.pt"

        t0 = time.time()
        model, best_val_f1, diagnostics, rgb_unchanged, final_epoch = fit(
            model, train_loader, val_loader, device,
            max_epochs=max_epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
            eta_min_factor=base_cfg["eta_min_factor"],
            head="corn", num_classes=num_classes,
            log_prefix=f"[{CONFIG_ID} fold{fold_idx}]",
            checkpoint_path=checkpoint_path, resume_path=resume_path, patience=patience,
        )
        train_time_s = time.time() - t0

        if not rgb_unchanged:
            print(f"[{CONFIG_ID} fold{fold_idx}] WARNING: RGB branch logits moved during training.", file=sys.stderr)

        from src.losses import build_loss  # noqa: E402
        criterion = build_loss("corn", num_classes)
        test_stats = run_epoch(model, test_loader, criterion, device, "corn", num_classes, optimizer=None)

        provenance = {
            "config_id": CONFIG_ID, "tissue_channel": "combined_max", "fold": fold_idx,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rgb_baseline_checkpoint": str(rgb_checkpoint_path.relative_to(REPO_ROOT)),
            "pretraining_verification": checks,
            "early_stopped_at_epoch": final_epoch, "max_epochs": max_epochs, "patience": patience,
            "rgb_logits_unchanged_across_training": rgb_unchanged,
            "params_m": round(params_m, 3), "lr": base_cfg["lr"], "seed": base_cfg["seed"],
        }
        with open(checkpoint_path.with_suffix(".provenance.json"), "w") as f:
            json.dump(provenance, f, indent=2)

        new_rows.append({
            "config_id": CONFIG_ID, "tissue_channel": "combined_max",
            "encoder": base_cfg["encoder"], "head": "corn", "weights": "baseline_checkpoint",
            "fold": fold_idx, "params_m": round(params_m, 3),
            "accuracy": round(test_stats["accuracy"], 4), "macro_f1": round(test_stats["macro_f1"], 4),
            "mae": round(test_stats["mae"], 4), "qwk": round(test_stats["qwk"], 4),
            "early_stopped_at_epoch": final_epoch, "train_time_s": round(train_time_s, 1),
        })
        print(f"[{CONFIG_ID} fold{fold_idx}] DONE test_acc={new_rows[-1]['accuracy']:.4f} "
              f"test_f1={new_rows[-1]['macro_f1']:.4f} test_qwk={new_rows[-1]['qwk']:.4f} "
              f"(stopped epoch {final_epoch}/{max_epochs})", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_merged_csv(results_csv, new_rows)
    print(f"[{CONFIG_ID}] wrote {len(new_rows)} row(s) to {results_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
