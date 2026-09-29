import argparse
import csv
import json
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.augmentations import IMAGENET_MEAN, IMAGENET_STD, build_resize
from src.corruptions import CORRUPTIONS
from src.dataset import WoundRegionDataset, list_filenames, make_kfold_splits, split_holdout
from src.metrics import confusion_from_logits, iou_dice_from_confusion
from src.model import build_model
from src.utils import load_color_mapping


def build_corrupted_transform(resolution, corruption_name):
    resize = build_resize(resolution)
    corrupt = CORRUPTIONS[corruption_name]
    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return A.Compose([*resize, corrupt, normalize, ToTensorV2()])


def evaluate_fold(ckpt_dir, case_index, fold_index, device):
    config = json.load(open(ckpt_dir / f"case_{case_index:02d}_fold{fold_index}_config.json"))

    all_filenames = list_filenames(config["data"]["images_dir"])
    dev_pool, _ = split_holdout(all_filenames, config["data"]["holdout_fraction"], config["data"]["holdout_seed"])
    folds = make_kfold_splits(
        dev_pool, config["data"]["n_folds"], config["data"]["split_seed"], config["data"]["val_fraction_of_remaining"]
    )
    _, _, test_files = folds[fold_index]

    classes = config["data"]["classes"]
    color_mapping = load_color_mapping(config["data"]["color_mapping_path"], classes)
    num_classes = len(classes)
    ignore_index = config["data"]["ignore_index"]

    model = build_model(config["model"]["decoder"], config["model"]["encoder"], config["model"]["weights"], num_classes)
    model.load_state_dict(torch.load(ckpt_dir / f"case_{case_index:02d}_fold{fold_index}.pt", map_location="cpu"))
    model.to(device)
    model.eval()

    fold_results = {}
    for corruption_name in CORRUPTIONS:
        tf = build_corrupted_transform(config["train"]["resolution"], corruption_name)
        ds = WoundRegionDataset(
            config["data"]["images_dir"], config["data"]["masks_dir"], test_files, classes, color_mapping, transform=tf
        )
        loader = DataLoader(ds, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=2)

        conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                conf += confusion_from_logits(logits, targets, num_classes, ignore_index)
        iou, dice = iou_dice_from_confusion(conf)
        fold_results[corruption_name] = (iou.mean().item(), dice.mean().item())

    return fold_results, config["data"]["n_folds"]


def main():
    parser = argparse.ArgumentParser(description="Test-time robustness sweep, averaged over all folds of a case")
    parser.add_argument("--checkpoint-dir", required=True, help="e.g. checkpoints/backbone")
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    fold_config_paths = sorted(ckpt_dir.glob(f"case_{args.case_index:02d}_fold*_config.json"))
    if not fold_config_paths:
        raise SystemExit(f"No checkpoints found for case {args.case_index} in {ckpt_dir}")
    n_folds = len(fold_config_paths)

    per_corruption = {name: {"miou": [], "mdice": []} for name in CORRUPTIONS}
    for fold_idx in range(n_folds):
        print(f"fold {fold_idx + 1}/{n_folds}")
        fold_results, _ = evaluate_fold(ckpt_dir, args.case_index, fold_idx, args.device)
        for corruption_name, (miou, mdice) in fold_results.items():
            per_corruption[corruption_name]["miou"].append(miou)
            per_corruption[corruption_name]["mdice"].append(mdice)
            print(f"  {corruption_name}: miou={miou:.4f} mdice={mdice:.4f}")

    rows = []
    for corruption_name, vals in per_corruption.items():
        miou_arr = np.array(vals["miou"])
        mdice_arr = np.array(vals["mdice"])
        rows.append({
            "corruption": corruption_name,
            "n_folds": n_folds,
            "miou_mean": float(miou_arr.mean()),
            "miou_std": float(miou_arr.std()),
            "mdice_mean": float(mdice_arr.mean()),
            "mdice_std": float(mdice_arr.std()),
        })

    out_csv = ckpt_dir / f"case_{args.case_index:02d}_robustness_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
