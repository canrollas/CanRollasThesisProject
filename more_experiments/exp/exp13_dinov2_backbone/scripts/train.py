#!/usr/bin/env python3
"""Exp13: DINOv2 ViT-L/14 (self-supervised, general-purpose, zero wound/
medical pretraining) as the wound-staging encoder, CORN head on top.

Two modes (see configs/base.yaml and README's "Fine-tune güvenliği"):
  --mode frozen    linear probe, only CornHead trains (~3K params)
  --mode finetune  all 303M params train, much smaller lr/batch/epochs

Same 3-fold split (task_3's src/splits.py build_folds, same seed) and
same image_size/augment as task_3's own ablation grid -- read directly
from task_3/configs/base.yaml, so the dataset side is byte-identical to
the mit_b1__corn / mit_b2__corn (ImageNet) reference rows. Only the
per-mode optimizer recipe (lr/batch/epochs) comes from this exp's own
config, since a 303M-param ViT needs a very different recipe than a
13-24M-param CNN/MiT encoder.

Downloads DINOv2's real pretrained weights (~1.2GB) on first run --
intended for Colab/GPU, not a laptop. Safe to interrupt and re-run --
already-completed (config_id, fold) rows are skipped, same convention as
task_3/run_ablation.py and exp12/scripts/train.py.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.augmentations import ImageAugmentation  # noqa: E402
from src.dataset import StageDataset  # noqa: E402
from src.engine import fit, run_epoch  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import append_result, get_device, load_completed_keys, load_yaml, set_seed, write_summary  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from model import build_model, count_params_m, count_trainable_params_m  # noqa: E402


def config_id(mode):
    return f"dinov2_vitl14__corn_{mode}"


def make_loaders(data_root, fold, recipe_cfg, batch_size, workers, smoke):
    aug = ImageAugmentation(**recipe_cfg["augment"])
    train_samples, val_samples, test_samples = fold["train"], fold["val"], fold["test"]
    if smoke:
        train_samples, val_samples, test_samples = train_samples[:8], val_samples[:4], test_samples[:4]

    train_ds = StageDataset(train_samples, data_root, recipe_cfg["image_size"], transform=aug)
    val_ds = StageDataset(val_samples, data_root, recipe_cfg["image_size"], transform=None)
    test_ds = StageDataset(test_samples, data_root, recipe_cfg["image_size"], transform=None)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["frozen", "finetune"])
    p.add_argument("--data-root", default=None, help="defaults to <repo>/datasets/wound-stage-classification")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--folds", default=None, help="comma-separated fold indices, e.g. '0,2' -- default: all")
    p.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs, tiny subset, pretrained=False -- sanity check")
    return p.parse_args()


def main():
    args = parse_args()
    exp_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    mode_cfg = exp_cfg["modes"][args.mode]
    recipe_cfg = load_yaml(REPO_ROOT / "task_3" / "configs" / "base.yaml")  # dataset/split recipe only

    data_root = Path(args.data_root) if args.data_root else REPO_ROOT / "datasets" / "wound-stage-classification"

    set_seed(recipe_cfg["seed"])
    folds = build_folds(data_root, recipe_cfg["num_folds"], recipe_cfg["val_fraction"], recipe_cfg["seed"])

    wanted_folds = set(int(x) for x in args.folds.split(",")) if args.folds else set(range(len(folds)))
    if args.smoke:
        folds = folds[:1]
        wanted_folds = {0}

    device = get_device()
    cid = config_id(args.mode)
    results_csv = EXP_DIR / exp_cfg["results_csv"]
    checkpoint_dir = EXP_DIR / exp_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(results_csv)

    epochs = args.epochs if args.epochs is not None else (2 if args.smoke else mode_cfg["epochs"])
    batch_size = args.batch if args.batch is not None else mode_cfg["batch_size"]
    num_classes = len(recipe_cfg["classes"])

    print(f"device={device} config={cid} folds={sorted(wanted_folds)} epochs={epochs} "
          f"lr={mode_cfg['lr']} batch={batch_size}", flush=True)

    for fold_idx, fold in enumerate(folds):
        if fold_idx not in wanted_folds or (cid, str(fold_idx)) in completed:
            continue

        train_loader, val_loader, test_loader = make_loaders(
            data_root, fold, recipe_cfg, batch_size, args.workers, args.smoke
        )

        model = build_model(args.mode, num_classes=num_classes, img_size=recipe_cfg["image_size"],
                             pretrained=not args.smoke)
        params_m = count_params_m(model)
        trainable_m = count_trainable_params_m(model)
        print(f"[fold{fold_idx}] params={params_m:.1f}M trainable={trainable_m:.4f}M "
              f"(mode={args.mode}, pretrained={not args.smoke})", flush=True)

        checkpoint_path = checkpoint_dir / f"{cid}_fold{fold_idx}.pt"
        resume_path = checkpoint_dir / f"{cid}_fold{fold_idx}.resume.pt"

        t0 = time.time()
        model, _ = fit(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=mode_cfg["lr"], weight_decay=mode_cfg["weight_decay"],
            eta_min_factor=exp_cfg["eta_min_factor"],
            head="corn", num_classes=num_classes,
            log_prefix=f"[{cid} fold{fold_idx}]",
            checkpoint_path=checkpoint_path, resume_path=resume_path,
        )
        train_time_s = time.time() - t0

        criterion = build_loss("corn", num_classes)
        test_stats = run_epoch(model, test_loader, criterion, device, "corn", num_classes, optimizer=None)

        row = {
            "config_id": cid, "encoder": "dinov2_vitl14", "head": "corn",
            "weights": f"dinov2_{args.mode}", "fold": fold_idx,
            "params_m": round(params_m, 3),
            "accuracy": round(test_stats["accuracy"], 4), "macro_f1": round(test_stats["macro_f1"], 4),
            "mae": round(test_stats["mae"], 4), "qwk": round(test_stats["qwk"], 4),
            "train_time_s": round(train_time_s, 1),
        }
        append_result(results_csv, row)
        write_summary(results_csv, results_csv.with_name("exp13_summary.csv"))
        print(f"[{cid} fold{fold_idx}] DONE test_acc={row['accuracy']:.4f} "
              f"test_f1={row['macro_f1']:.4f} test_qwk={row['qwk']:.4f}", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
