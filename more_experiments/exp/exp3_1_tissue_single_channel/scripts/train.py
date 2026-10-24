#!/usr/bin/env python3
"""Experiment 3.1 (Chapter 8): staging on RGB + 1 severity-banded tissue channel.

Single config -- MiT-B1 + CORN, ImageNet weights, matching task_3's and
exp-3's confirmed best config -- trained on
datasets/wound-staging-masks-exp-3-1/ (full uncropped RGB + 1 severity
channel collapsing exp-3's 5 Task-2-derived tissue-probability channels,
4 channels total; see ../scripts/generate_dataset.py).

Reuses task_3/src's engine/losses/metrics/splits/augmentations/utils
unchanged -- those only look at logits/targets/sample lists, not input
channel count or file format. model.py and dataset.py are local (4-channel
backbone with an EXPLICIT first-conv expansion -- pretrained RGB weights
copied unchanged into channels 0-2, the severity channel 3 zero-
initialized, so the model is mathematically identical to the RGB-only
baseline before any training happens; see model.py's docstring and
scripts/test_4ch_model.py for the verification. .npy loader with separate
RGB/tissue normalization) since task_3/src's own versions hardcode
3-channel JPEGs and exp-3's hardcode 5 tissue channels.

Run scripts/test_4ch_model.py before trusting any training run from this
script -- it checks the zero-init property, gradient flow into the
severity channel, and output shapes directly rather than assuming them.

build_folds(data_root, ...) run against wound-staging-masks-exp-3-1
reproduces the SAME fold assignment as the original Task 3 ablation and
exp-3 (same filenames -- .npy extension doesn't affect sort order,
verified directly) for a paired comparison against exp-3 and Experiment
1A/1B/2.

Safe to interrupt and resume, same pattern as the other exp/*/scripts/train.py.
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

THIS_DIR = Path(__file__).resolve().parent  # exp/exp3_1_tissue_single_channel/scripts/
EXP_DIR = THIS_DIR.parent  # exp/exp3_1_tissue_single_channel/ -- configs/checkpoints/results live here
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.augmentations import ImageAugmentation  # noqa: E402
from src.engine import fit, run_epoch  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import (  # noqa: E402
    append_failure, append_result, get_device, load_completed_keys, load_yaml, set_seed, write_summary,
)

sys.path.insert(0, str(THIS_DIR))
from dataset import TissueChannelStageDataset  # noqa: E402
from model import build_model, count_params_m  # noqa: E402

CONFIG_ID = "mit_b1__corn__exp3_1_tissue_single_channel"


def make_loaders(data_root, fold, base_cfg, batch_size, workers, smoke):
    aug = ImageAugmentation(**base_cfg["augment"])
    train_samples, val_samples, test_samples = fold["train"], fold["val"], fold["test"]

    if smoke:
        train_samples, val_samples, test_samples = train_samples[:8], val_samples[:4], test_samples[:4]

    train_ds = TissueChannelStageDataset(train_samples, data_root, base_cfg["image_size"], transform=aug)
    val_ds = TissueChannelStageDataset(val_samples, data_root, base_cfg["image_size"], transform=None)
    test_ds = TissueChannelStageDataset(test_samples, data_root, base_cfg["image_size"], transform=None)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None, help="defaults to configs/base.yaml's data_root (wound-staging-masks-exp-3-1)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs, tiny subset -- sanity check")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")

    data_root = Path(args.data_root) if args.data_root else EXP_DIR / base_cfg["data_root"]

    set_seed(base_cfg["seed"])
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    if args.smoke:
        folds = folds[:1]

    device = get_device()
    results_csv = EXP_DIR / base_cfg["results_csv"]
    failures_csv = results_csv.with_name("exp3_1_failures.csv")
    checkpoint_dir = EXP_DIR / base_cfg["checkpoint_dir"]
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
                data_root, fold, base_cfg, batch_size, args.workers, args.smoke
            )

            model = build_model(base_cfg["encoder"], base_cfg["weights"], base_cfg["head"],
                                 num_classes=num_classes, in_channels=base_cfg["in_channels"])
            params_m = count_params_m(model)
            checkpoint_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.pt"
            resume_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.resume.pt"

            t0 = time.time()
            model, _ = fit(
                model, train_loader, val_loader, device,
                epochs=epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
                eta_min_factor=base_cfg["eta_min_factor"],
                head=base_cfg["head"], num_classes=num_classes,
                log_prefix=f"[{CONFIG_ID} fold{fold_idx}]",
                checkpoint_path=checkpoint_path,
                resume_path=resume_path,
            )
            train_time_s = time.time() - t0

            provenance_path = checkpoint_path.with_suffix(".provenance.json")
            with open(provenance_path, "w") as f:
                json.dump({
                    "config_id": CONFIG_ID,
                    "fold": fold_idx,
                    "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "encoder": base_cfg["encoder"],
                    "head": base_cfg["head"],
                    "in_channels": base_cfg["in_channels"],
                    "pretrained_weights_source": f"segmentation_models_pytorch encoder_weights={base_cfg['weights']!r}",
                    "first_conv_init": (
                        "RGB channels (0-2) copied unchanged from the pretrained 3-channel "
                        "encoder's first conv; severity channel (3) zero-initialized -- see model.py"
                    ),
                    "image_size": base_cfg["image_size"],
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": base_cfg["lr"],
                    "weight_decay": base_cfg["weight_decay"],
                    "eta_min_factor": base_cfg["eta_min_factor"],
                    "seed": base_cfg["seed"],
                    "data_root": str(data_root),
                    "params_m": round(params_m, 3),
                }, f, indent=2)

            criterion = build_loss(base_cfg["head"], num_classes)
            test_stats = run_epoch(model, test_loader, criterion, device, base_cfg["head"], num_classes,
                                    optimizer=None)

            row = {
                "config_id": CONFIG_ID,
                "encoder": base_cfg["encoder"],
                "head": base_cfg["head"],
                "weights": base_cfg["weights"],
                "fold": fold_idx,
                "params_m": round(params_m, 3),
                "accuracy": round(test_stats["accuracy"], 4),
                "macro_f1": round(test_stats["macro_f1"], 4),
                "mae": round(test_stats["mae"], 4),
                "qwk": round(test_stats["qwk"], 4),
                "train_time_s": round(train_time_s, 1),
            }
            append_result(results_csv, row)
            write_summary(results_csv, results_csv.with_name("exp3_1_summary.csv"))
            print(f"[{CONFIG_ID} fold{fold_idx}] DONE test_acc={row['accuracy']:.4f} "
                  f"test_f1={row['macro_f1']:.4f} test_qwk={row['qwk']:.4f}", flush=True)

        except Exception as exc:  # noqa: BLE001 - keep going so a bad fold doesn't lose the others
            traceback.print_exc()
            print(f"[{CONFIG_ID} fold{fold_idx}] FAILED: {exc}", flush=True)
            append_failure(failures_csv, {
                "config_id": CONFIG_ID,
                "encoder": base_cfg["encoder"],
                "head": base_cfg["head"],
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
