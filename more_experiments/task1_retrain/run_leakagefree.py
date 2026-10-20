#!/usr/bin/env python3
"""Leakage-free Task 1 retrain (Chapter 8, Step 1).

Trains a single config -- MAnet + MiT-B2, ImageNet weights, matching the
original Task 1 best config -- on the 3,248 images that remain after
excluding every PIID image (see ../manifests/piid_overlap_report.json and
leakagefree_splits.py). This checkpoint is the only Task-1 model allowed to
generate localisation info for the rest of Chapter 8.

Reuses task_1/src training code verbatim (model/losses/engine/metrics/
augmentations/visualize/utils) -- only the file list and fold construction
differ. task_1/ itself is untouched; this writes to
./checkpoints and ./results only.

Safe to interrupt and resume: completed (config, fold) rows are skipped on
the results CSV, and each fold checkpoints every epoch via the same
resume_path mechanism as task_1/run_ablation.py.
"""
import argparse
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "task_1"))

from src.augmentations import PairedAugmentation  # noqa: E402
from src.dataset import WoundRegionDataset, load_color_mapping  # noqa: E402
from src.efficiency import count_params_m  # noqa: E402
from src.engine import fit, run_epoch  # noqa: E402
from src.losses import CombinedLoss  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import (  # noqa: E402
    append_failure, append_result, get_device, load_completed_keys, load_yaml, set_seed, write_summary,
)
from src.visualize import save_sample_panel  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from leakagefree_splits import build_folds  # noqa: E402

CONFIG_ID = "manet__mit_b2__imagenet_leakagefree"


def make_loaders(data_root, fold, base_cfg, colors, batch_size, workers, smoke):
    aug = PairedAugmentation(**base_cfg["augment"])
    train_files, val_files, test_files = fold["train"], fold["val"], fold["test"]

    if smoke:
        train_files, val_files, test_files = train_files[:8], val_files[:4], test_files[:4]

    train_ds = WoundRegionDataset(train_files, data_root, base_cfg["image_size"], colors, transform=aug)
    val_ds = WoundRegionDataset(val_files, data_root, base_cfg["image_size"], colors, transform=None)
    test_ds = WoundRegionDataset(test_files, data_root, base_cfg["image_size"], colors, transform=None)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None, help="defaults to <repo>/datasets/wound-region-segmentation")
    p.add_argument("--filelist", default=None, help="defaults to ../manifests/task1_leakagefree_filelist.txt")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs, tiny subset -- sanity check")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(THIS_DIR / "configs" / "base.yaml")

    repo_root = REPO_ROOT
    data_root = Path(args.data_root) if args.data_root else repo_root / "datasets" / "wound-region-segmentation"
    filelist = Path(args.filelist) if args.filelist else THIS_DIR / base_cfg["filelist"]
    colors = load_color_mapping(data_root)

    set_seed(base_cfg["seed"])
    folds = build_folds(filelist, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    if args.smoke:
        folds = folds[:1]

    device = get_device()
    results_csv = THIS_DIR / base_cfg["results_csv"]
    failures_csv = results_csv.with_name("leakagefree_failures.csv")
    checkpoint_dir = THIS_DIR / base_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(results_csv)

    epochs = args.epochs if args.epochs is not None else (2 if args.smoke else base_cfg["epochs"])
    batch_size = args.batch if args.batch is not None else base_cfg["batch_size"]
    num_classes = len(base_cfg["classes"])

    print(
        f"device={device} config={CONFIG_ID} folds={len(folds)} epochs={epochs} "
        f"train_size={len(folds[0]['train'])} val_size={len(folds[0]['val'])} test_size={len(folds[0]['test'])}",
        flush=True,
    )

    for fold_idx, fold in enumerate(folds):
        if (CONFIG_ID, str(fold_idx)) in completed:
            print(f"[{CONFIG_ID} fold{fold_idx}] already completed, skipping", flush=True)
            continue

        model = None
        try:
            train_loader, val_loader, test_loader = make_loaders(
                data_root, fold, base_cfg, colors, batch_size, args.workers, args.smoke
            )

            model = build_model(base_cfg["architecture"], base_cfg["encoder"], base_cfg["weights"], num_classes)
            params_m = count_params_m(model)
            checkpoint_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.pt"
            resume_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.resume.pt"

            t0 = time.time()
            model, _ = fit(
                model, train_loader, val_loader, device,
                epochs=epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
                eta_min_factor=base_cfg["eta_min_factor"],
                ce_weight=base_cfg["loss"]["ce_weight"], dice_weight=base_cfg["loss"]["dice_weight"],
                num_classes=num_classes,
                log_prefix=f"[{CONFIG_ID} fold{fold_idx}]",
                checkpoint_path=checkpoint_path,
                resume_path=resume_path,
            )
            train_time_s = time.time() - t0

            criterion = CombinedLoss(num_classes=num_classes,
                                      ce_weight=base_cfg["loss"]["ce_weight"],
                                      dice_weight=base_cfg["loss"]["dice_weight"])
            test_stats = run_epoch(model, test_loader, criterion, device, optimizer=None, num_classes=num_classes)

            try:
                sample_idx = random.randrange(len(test_loader.dataset))
                sample_image, sample_gt = test_loader.dataset[sample_idx]
                model.eval()
                with torch.no_grad():
                    logits = model(sample_image.unsqueeze(0).to(device))
                    pred_mask = logits.argmax(dim=1).squeeze(0).cpu()
                sample_path = results_csv.parent / "samples" / f"{CONFIG_ID}_fold{fold_idx}.png"
                save_sample_panel(sample_image, sample_gt, pred_mask, colors, sample_path)
            except Exception as exc:  # noqa: BLE001 - a bad sample image shouldn't kill the run
                print(f"[{CONFIG_ID} fold{fold_idx}] sample panel failed: {exc}", flush=True)

            row = {
                "config_id": CONFIG_ID,
                "architecture": base_cfg["architecture"],
                "encoder": base_cfg["encoder"],
                "weights": base_cfg["weights"],
                "fold": fold_idx,
                "params_m": round(params_m, 3),
                "miou": round(test_stats["miou"], 4),
                "mdice": round(test_stats["mdice"], 4),
                "iou_bg": round(test_stats["iou_per_class"][0], 4),
                "iou_skin": round(test_stats["iou_per_class"][1], 4),
                "iou_wound": round(test_stats["iou_per_class"][2], 4),
                "dice_bg": round(test_stats["dice_per_class"][0], 4),
                "dice_skin": round(test_stats["dice_per_class"][1], 4),
                "dice_wound": round(test_stats["dice_per_class"][2], 4),
                "train_time_s": round(train_time_s, 1),
            }
            append_result(results_csv, row)
            write_summary(results_csv, results_csv.with_name("leakagefree_summary.csv"))
            print(f"[{CONFIG_ID} fold{fold_idx}] DONE test_miou={row['miou']:.4f}", flush=True)

        except Exception as exc:  # noqa: BLE001 - keep going so a bad fold doesn't lose the others
            traceback.print_exc()
            print(f"[{CONFIG_ID} fold{fold_idx}] FAILED: {exc}", flush=True)
            append_failure(failures_csv, {
                "config_id": CONFIG_ID,
                "architecture": base_cfg["architecture"],
                "encoder": base_cfg["encoder"],
                "weights": base_cfg["weights"],
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
