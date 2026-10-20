#!/usr/bin/env python3
"""Experiment 4 (Chapter 8): late fusion of RGB and tissue-probability features.

RGB (ImageNet MiT-B1) and tissue (small CNN) are encoded separately, each
pooled to a feature vector, combined by --fusion (concat MLP | gated
residual -- see scripts/model.py), then classified by the same CornHead
task_3 uses everywhere else. Trained on the same data as exp-3
(datasets/wound-staging-masks-exp-3/, split into two streams by
scripts/dataset.py) and the same verified PIID 3-fold split.

engine.py is a local fork of task_3/src/engine.py (model(rgb, tissue)
instead of model(images), plus optional staged freeze/fine-tune) --
losses/metrics/splits/augmentations/utils are task_3/src's own, unchanged.

Writes two rows per run (one per --fusion value) to the same results CSV,
config_id-differentiated, same pattern task_3's own ablation uses for
multiple configs sharing one file.

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

THIS_DIR = Path(__file__).resolve().parent  # exp/exp4_late_fusion/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.augmentations import ImageAugmentation  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import (  # noqa: E402
    append_failure, append_result, get_device, load_completed_keys, load_yaml, set_seed, write_summary,
)

sys.path.insert(0, str(THIS_DIR))
from dataset import TwoStreamStageDataset  # noqa: E402
from engine import fit, run_epoch  # noqa: E402
from model import build_model, count_params_m  # noqa: E402


def make_loaders(data_root, fold, base_cfg, batch_size, workers, smoke):
    aug = ImageAugmentation(**base_cfg["augment"])
    train_samples, val_samples, test_samples = fold["train"], fold["val"], fold["test"]

    if smoke:
        train_samples, val_samples, test_samples = train_samples[:8], val_samples[:4], test_samples[:4]

    train_ds = TwoStreamStageDataset(train_samples, data_root, base_cfg["image_size"], transform=aug)
    val_ds = TwoStreamStageDataset(val_samples, data_root, base_cfg["image_size"], transform=None)
    test_ds = TwoStreamStageDataset(test_samples, data_root, base_cfg["image_size"], transform=None)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fusion", default=None, choices=["concat", "gated"],
                    help="defaults to configs/base.yaml's fusion field")
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--freeze-epochs", type=int, default=None,
                    help="defaults to configs/base.yaml's freeze_epochs (0 = no staged freezing)")
    p.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs, tiny subset -- sanity check")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")

    fusion = args.fusion or base_cfg["fusion"]
    freeze_epochs = args.freeze_epochs if args.freeze_epochs is not None else base_cfg["freeze_epochs"]
    config_id = f"mit_b1__corn__exp4_latefusion_{fusion}"
    if freeze_epochs > 0:
        config_id += f"_freeze{freeze_epochs}"

    data_root = Path(args.data_root) if args.data_root else EXP_DIR / base_cfg["data_root"]

    set_seed(base_cfg["seed"])
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    if args.smoke:
        folds = folds[:1]

    device = get_device()
    results_csv = EXP_DIR / base_cfg["results_csv"]
    failures_csv = results_csv.with_name("exp4_failures.csv")
    checkpoint_dir = EXP_DIR / base_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(results_csv)

    epochs = args.epochs if args.epochs is not None else (2 if args.smoke else base_cfg["epochs"])
    batch_size = args.batch if args.batch is not None else base_cfg["batch_size"]
    num_classes = len(base_cfg["classes"])

    print(
        f"device={device} config={config_id} folds={len(folds)} epochs={epochs} freeze_epochs={freeze_epochs} "
        f"train_size={len(folds[0]['train'])} val_size={len(folds[0]['val'])} test_size={len(folds[0]['test'])}",
        flush=True,
    )

    for fold_idx, fold in enumerate(folds):
        if (config_id, str(fold_idx)) in completed:
            print(f"[{config_id} fold{fold_idx}] already completed, skipping", flush=True)
            continue

        model = None
        try:
            train_loader, val_loader, test_loader = make_loaders(
                data_root, fold, base_cfg, batch_size, args.workers, args.smoke
            )

            model = build_model(base_cfg["encoder"], base_cfg["weights"], fusion, num_classes=num_classes)
            params_m = count_params_m(model)
            checkpoint_path = checkpoint_dir / f"{config_id}_fold{fold_idx}.pt"
            resume_path = checkpoint_dir / f"{config_id}_fold{fold_idx}.resume.pt"

            t0 = time.time()
            model, _ = fit(
                model, train_loader, val_loader, device,
                epochs=epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
                eta_min_factor=base_cfg["eta_min_factor"],
                head="corn", num_classes=num_classes,
                log_prefix=f"[{config_id} fold{fold_idx}]",
                checkpoint_path=checkpoint_path,
                resume_path=resume_path,
                freeze_epochs=freeze_epochs,
                finetune_lr=base_cfg.get("finetune_lr"),
            )
            train_time_s = time.time() - t0

            provenance_path = checkpoint_path.with_suffix(".provenance.json")
            with open(provenance_path, "w") as f:
                json.dump({
                    "config_id": config_id, "fold": fold_idx,
                    "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "architecture": "late fusion: ImageNet MiT-B1 (RGB) + small CNN (tissue), CORN head",
                    "fusion": fusion,
                    "rgb_arm_init": "ImageNet pretrained only -- no staging-finetuned RGB checkpoint existed locally",
                    "freeze_epochs": freeze_epochs,
                    "finetune_lr": base_cfg.get("finetune_lr") if freeze_epochs > 0 else None,
                    "image_size": base_cfg["image_size"], "batch_size": batch_size, "epochs": epochs,
                    "lr": base_cfg["lr"], "weight_decay": base_cfg["weight_decay"], "seed": base_cfg["seed"],
                    "data_root": str(data_root), "params_m": round(params_m, 3),
                }, f, indent=2)

            criterion = build_loss("corn", num_classes)
            test_stats = run_epoch(model, test_loader, criterion, device, "corn", num_classes, optimizer=None)

            row = {
                "config_id": config_id,
                "encoder": base_cfg["encoder"],
                "head": "corn",
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
            write_summary(results_csv, results_csv.with_name("exp4_summary.csv"))
            print(f"[{config_id} fold{fold_idx}] DONE test_acc={row['accuracy']:.4f} "
                  f"test_f1={row['macro_f1']:.4f} test_qwk={row['qwk']:.4f}", flush=True)

        except Exception as exc:  # noqa: BLE001 - keep going so a bad fold doesn't lose the others
            traceback.print_exc()
            print(f"[{config_id} fold{fold_idx}] FAILED: {exc}", flush=True)
            append_failure(failures_csv, {
                "config_id": config_id,
                "encoder": base_cfg["encoder"],
                "head": "corn",
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
