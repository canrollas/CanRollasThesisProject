#!/usr/bin/env python3
"""Re-evaluates the 10 already-trained Task 1 checkpoints on their held-out
test folds to add precision/recall per class and the full confusion matrix,
none of which run_ablation.py computed or saved (it only kept IoU/Dice).
Does not retrain anything; only loads each saved checkpoint and runs
inference, using the exact same fold split (same seed) so each checkpoint
sees the same test files it was evaluated on originally.

Also recomputes IoU/Dice from the same confusion matrices as a sanity check
against results/ablation_summary.csv -- if those don't match, something about
this script's reconstruction of the eval pipeline is wrong.

Writes two CSVs incrementally (one row appended per config, as soon as that
config's 3 folds are done), so a Colab disconnect only loses the config in
progress, not everything before it:
  - results/precision_recall.csv   -- precision/recall per class, mean+std over folds
  - results/confusion_matrices.csv -- summed 3x3 confusion matrix (pixel counts) over all 3 folds
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR))

from src.dataset import WoundRegionDataset, load_color_mapping  # noqa: E402
from src.metrics import compute_confusion, iou_dice_from_confusion  # noqa: E402
from src.model import build_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402

CLASS_NAMES = ["bg", "skin", "wound"]

PR_CSV = "precision_recall.csv"
CONF_CSV = "confusion_matrices.csv"


def precision_recall_from_confusion(conf):
    conf = conf.astype("float64")
    tp = conf.diagonal()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    precision = tp / (tp + fp).clip(min=1e-9)
    recall = tp / (tp + fn).clip(min=1e-9)
    return precision, recall


def load_done_config_ids(path):
    if not path.exists():
        return set()
    with open(path) as f:
        reader = csv.DictReader(f)
        return {row["config_id"] for row in reader}


def append_row(path, row):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    base_cfg = load_yaml(TASK_DIR / "configs" / "base.yaml")
    grid = json.loads((TASK_DIR / "configs" / "resolved_grid.json").read_text())
    by_id = {c["config_id"]: c for c in grid}

    summary_path = TASK_DIR / "results" / "task_1_outputs" / "results" / "ablation_summary.csv"
    summary_config_ids = []
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary_config_ids.append(row["config_id"])

    repo_root = TASK_DIR.parent
    data_root = repo_root / "datasets" / "wound-region-segmentation"
    colors = load_color_mapping(data_root)
    num_classes = len(base_cfg["classes"])
    device = get_device()

    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    checkpoint_dir = TASK_DIR / "results" / "task_1_outputs" / "checkpoints"
    results_dir = TASK_DIR / "results" / "task_1_outputs" / "results"

    pr_path = results_dir / PR_CSV
    conf_path = results_dir / CONF_CSV
    done = load_done_config_ids(pr_path)

    for config_id in summary_config_ids:
        if config_id in done:
            print(f"skip (already done): {config_id}")
            continue

        cfg = by_id[config_id]
        per_fold_precision = []
        per_fold_recall = []
        per_fold_miou = []
        per_fold_mdice = []
        conf_total = np.zeros((num_classes, num_classes), dtype=np.int64)

        for fold_idx, fold in enumerate(folds):
            ckpt_path = checkpoint_dir / f"{config_id}_fold{fold_idx}.pt"
            if not ckpt_path.exists():
                print(f"MISSING checkpoint: {ckpt_path}", file=sys.stderr)
                continue

            test_ds = WoundRegionDataset(fold["test"], data_root, base_cfg["image_size"], colors, transform=None)
            test_loader = DataLoader(test_ds, batch_size=base_cfg["batch_size"], shuffle=False, num_workers=2)

            model = build_model(cfg["architecture"], cfg["encoder"], None, num_classes=num_classes)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()

            conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
            with torch.no_grad():
                for images, targets in test_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    logits = model(images)
                    conf += compute_confusion(logits, targets, num_classes)

            conf_np = conf.cpu().numpy()
            conf_total += conf_np
            precision, recall = precision_recall_from_confusion(conf_np)
            iou, dice = iou_dice_from_confusion(conf_np)

            per_fold_precision.append(precision)
            per_fold_recall.append(recall)
            per_fold_miou.append(iou.mean())
            per_fold_mdice.append(dice.mean())

            print(f"{config_id} fold{fold_idx}: miou={iou.mean():.4f} mdice={dice.mean():.4f} "
                  f"precision={np.round(precision, 4)} recall={np.round(recall, 4)}", flush=True)

        per_fold_precision = np.stack(per_fold_precision)
        per_fold_recall = np.stack(per_fold_recall)
        pr_row = {
            "config_id": config_id,
            "miou_mean_recomputed": float(np.mean(per_fold_miou)),
            "mdice_mean_recomputed": float(np.mean(per_fold_mdice)),
        }
        for i, name in enumerate(CLASS_NAMES):
            pr_row[f"precision_{name}_mean"] = float(per_fold_precision[:, i].mean())
            pr_row[f"precision_{name}_std"] = float(per_fold_precision[:, i].std())
            pr_row[f"recall_{name}_mean"] = float(per_fold_recall[:, i].mean())
            pr_row[f"recall_{name}_std"] = float(per_fold_recall[:, i].std())
        append_row(pr_path, pr_row)

        conf_row = {"config_id": config_id}
        for i, true_name in enumerate(CLASS_NAMES):
            for j, pred_name in enumerate(CLASS_NAMES):
                conf_row[f"true_{true_name}_pred_{pred_name}"] = int(conf_total[i, j])
        append_row(conf_path, conf_row)

        print(f"-- wrote row for {config_id} --\n", flush=True)

    print(f"\ndone. see {pr_path} and {conf_path}")


if __name__ == "__main__":
    main()
