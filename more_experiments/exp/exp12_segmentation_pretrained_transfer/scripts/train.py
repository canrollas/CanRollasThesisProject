#!/usr/bin/env python3
"""Exp12: Task-2-pretrained-encoder transfer for CORN wound staging.

Takes Task 2's own MAnet+MiT-B2 encoder (trained to separate granulation /
slough / necrosis / tendon / bone at the pixel level, on a dataset with
zero PIID overlap with the staging dataset -- see README's "Leakage"
section), discards the segmentation decoder, attaches a fresh CORN
ordinal head, and fully fine-tunes on the PIID staging dataset. Same
3-fold split (src/splits.py's build_folds, same seed) and same training
recipe (lr/epochs/batch/image_size/augment) as task_3's own mit_b2__corn
(ImageNet-init) row in task_3/results/task_3_outputs/results/
ablation_results.csv -- read directly from task_3/configs/base.yaml here,
so there is no risk of silent hyperparameter drift between the two.

Fold <-> Task-2-checkpoint pairing is arbitrary (task_3 fold f uses Task
2's wound_inner_fold{f+1} encoder) since the two datasets share no
PIID/case -- see README. Each of Task 2's 3 independently-trained fold
encoders is used exactly once, instead of reusing a single one for all 3
task_3 folds.

Safe to interrupt and re-run -- already-completed (config_id, fold) rows
are skipped, same convention as task_3/run_ablation.py. Does not write to
task_3/ or any other more_experiments/exp/*/ -- only ./checkpoints and
./results here.
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
from src.engine import fit, run_epoch  # noqa: E402
from src.dataset import StageDataset  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import append_result, get_device, load_completed_keys, load_yaml, set_seed, write_summary  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from model import build_model  # noqa: E402

CONFIG_ID = "mit_b2_task2init__corn"
ENCODER_NAME = "mit_b2"


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
    p.add_argument("--data-root", default=None, help="defaults to <repo>/datasets/wound-stage-classification")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--folds", default=None, help="comma-separated fold indices, e.g. '0,2' -- default: all")
    p.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs, tiny subset -- sanity check")
    return p.parse_args()


def main():
    args = parse_args()
    exp_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    # Shared training recipe lives in task_3's OWN config, not duplicated here -- see module docstring.
    recipe_cfg = load_yaml(REPO_ROOT / "task_3" / "configs" / "base.yaml")

    data_root = Path(args.data_root) if args.data_root else REPO_ROOT / "datasets" / "wound-stage-classification"

    set_seed(recipe_cfg["seed"])
    folds = build_folds(data_root, recipe_cfg["num_folds"], recipe_cfg["val_fraction"], recipe_cfg["seed"])

    wanted_folds = set(int(x) for x in args.folds.split(",")) if args.folds else set(range(len(folds)))
    if args.smoke:
        folds = folds[:1]
        wanted_folds = {0}

    device = get_device()
    results_csv = EXP_DIR / exp_cfg["results_csv"]
    checkpoint_dir = EXP_DIR / exp_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(results_csv)

    epochs = args.epochs if args.epochs is not None else (2 if args.smoke else recipe_cfg["epochs"])
    batch_size = args.batch if args.batch is not None else recipe_cfg["batch_size"]
    num_classes = len(recipe_cfg["classes"])

    print(f"device={device} config={CONFIG_ID} folds={sorted(wanted_folds)} epochs={epochs}", flush=True)

    for fold_idx, fold in enumerate(folds):
        if fold_idx not in wanted_folds or (CONFIG_ID, str(fold_idx)) in completed:
            continue

        task2_fold = fold_idx + 1  # see README/config: arbitrary-but-fixed pairing, no shared PIID either way
        task2_checkpoint_path = (
            REPO_ROOT / exp_cfg["task2_checkpoint_template"].format(task2_fold=task2_fold)
        ).resolve()
        if not task2_checkpoint_path.is_file():
            raise SystemExit(
                f"\nBLOCKED: Task 2 encoder checkpoint not found for fold {fold_idx}:\n"
                f"  {task2_checkpoint_path}\n"
                f"This experiment needs Task 2's own trained MAnet_mit_b2_imagenet.pth "
                f"(wound_inner_fold{task2_fold}) -- not retrained here."
            )

        train_loader, val_loader, test_loader = make_loaders(
            data_root, fold, recipe_cfg, batch_size, args.workers, args.smoke
        )

        model, verify = build_model(task2_checkpoint_path, device, num_classes=num_classes, encoder_name=ENCODER_NAME)
        print(
            f"[fold{fold_idx}] loaded Task2 encoder from {task2_checkpoint_path.name} "
            f"({verify['checkpoint_encoder_keys']}/{verify['backbone_encoder_keys']} keys, "
            f"changed_from_init={verify['weights_changed_from_init']})",
            flush=True,
        )
        if (
            verify["checkpoint_encoder_keys"] != verify["backbone_encoder_keys"]
            or not verify["weights_changed_from_init"]
        ):
            raise SystemExit(f"[fold{fold_idx}] STOP: encoder weight transfer looks wrong, see verify dict above.")

        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        checkpoint_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.pt"
        resume_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.resume.pt"

        t0 = time.time()
        model, _ = fit(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=recipe_cfg["lr"], weight_decay=recipe_cfg["weight_decay"],
            eta_min_factor=recipe_cfg["eta_min_factor"],
            head="corn", num_classes=num_classes,
            log_prefix=f"[{CONFIG_ID} fold{fold_idx}]",
            checkpoint_path=checkpoint_path, resume_path=resume_path,
        )
        train_time_s = time.time() - t0

        criterion = build_loss("corn", num_classes)
        test_stats = run_epoch(model, test_loader, criterion, device, "corn", num_classes, optimizer=None)

        row = {
            "config_id": CONFIG_ID, "encoder": ENCODER_NAME, "head": "corn",
            "weights": f"task2_manet_encoder_fold{task2_fold}", "fold": fold_idx,
            "params_m": round(params_m, 3),
            "accuracy": round(test_stats["accuracy"], 4), "macro_f1": round(test_stats["macro_f1"], 4),
            "mae": round(test_stats["mae"], 4), "qwk": round(test_stats["qwk"], 4),
            "train_time_s": round(train_time_s, 1),
        }
        append_result(results_csv, row)
        write_summary(results_csv, results_csv.with_name("exp12_summary.csv"))
        print(
            f"[{CONFIG_ID} fold{fold_idx}] DONE test_acc={row['accuracy']:.4f} "
            f"test_f1={row['macro_f1']:.4f} test_qwk={row['qwk']:.4f}",
            flush=True,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
