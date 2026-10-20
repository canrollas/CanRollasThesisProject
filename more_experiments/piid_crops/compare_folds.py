#!/usr/bin/env python3
"""Compares two generate_masks.py outputs (e.g. fold0 vs fold1) run on the
same staging dataset.

There's no ground-truth wound-region mask for the staging dataset (PIID
only has stage labels, not segmentation masks), so this can't say which
fold is "more correct" in isolation. What it *can* do:

  - per-stage detection-rate agreement (do both folds find/miss a wound on
    the same images?)
  - per-image wound-mask IoU between the two folds' predictions (high
    agreement = both checkpoints localise the same region = safer to use
    either for downstream crops; low agreement flags images worth a manual
    look)

For the actual "which fold's segmentation model is more accurate" answer,
use each fold's own held-out test mIoU/mDice from
task1_retrain/results/leakagefree_results.csv (3-fold CV on Task 1's own
leakage-free data) -- that's the number with real ground truth behind it.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

STAGE_DIRS = ["stage_1", "stage_2", "stage_3", "stage_4"]
WOUND_RGB = (255, 0, 0)  # #FF0000, from datasets/wound-region-segmentation/color-mappings.json


def load_metadata(output_dir):
    rows = {}
    with open(Path(output_dir) / "metadata.csv") as f:
        for row in csv.DictReader(f):
            rows[row["filename"]] = row
    return rows


def wound_mask(png_path):
    arr = np.array(Image.open(png_path).convert("RGB"))
    return np.all(arr == np.array(WOUND_RGB, dtype=np.uint8), axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="first generate_masks.py output dir (e.g. fold0)")
    p.add_argument("--b", required=True, help="second generate_masks.py output dir (e.g. fold1)")
    p.add_argument("--label-a", default="a")
    p.add_argument("--label-b", default="b")
    args = p.parse_args()

    meta_a = load_metadata(args.a)
    meta_b = load_metadata(args.b)
    common = sorted(set(meta_a) & set(meta_b))
    if len(common) != len(meta_a) or len(common) != len(meta_b):
        print(f"WARNING: {len(meta_a)} files in {args.label_a}, {len(meta_b)} in {args.label_b}, "
              f"{len(common)} in common -- comparing only the common set")

    print(f"\n=== Detection rate by stage ({args.label_a} vs {args.label_b}) ===")
    for stage in STAGE_DIRS:
        stage_files = [f for f in common if meta_a[f]["stage"] == stage]
        if not stage_files:
            continue
        det_a = sum(1 for f in stage_files if meta_a[f]["detected"] == "True")
        det_b = sum(1 for f in stage_files if meta_b[f]["detected"] == "True")
        both = sum(1 for f in stage_files if meta_a[f]["detected"] == "True" and meta_b[f]["detected"] == "True")
        neither = sum(1 for f in stage_files if meta_a[f]["detected"] == "False" and meta_b[f]["detected"] == "False")
        disagree = len(stage_files) - both - neither
        print(f"{stage}: n={len(stage_files)}  {args.label_a}={det_a}  {args.label_b}={det_b}  "
              f"both_detect={both}  both_miss={neither}  disagree={disagree}")

    print(f"\n=== Per-image wound-mask IoU ({args.label_a} vs {args.label_b}) ===")
    ious = []
    worst = []
    for f in common:
        stage = meta_a[f]["stage"]
        stem = Path(f).stem + ".png"
        mask_a = wound_mask(Path(args.a) / stage / "masks" / stem)
        mask_b = wound_mask(Path(args.b) / stage / "masks" / stem)
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        iou = 1.0 if union == 0 else inter / union
        ious.append(iou)
        worst.append((iou, f))

    ious = np.array(ious)
    print(f"mean IoU: {ious.mean():.4f}  median: {np.median(ious):.4f}  "
          f"min: {ious.min():.4f}  max: {ious.max():.4f}")
    print(f"images with IoU < 0.5: {(ious < 0.5).sum()} / {len(ious)}")

    worst.sort()
    print("\nlowest-agreement images (fold disagrees most):")
    for iou, f in worst[:10]:
        print(f"  {f}  IoU={iou:.4f}  wound_frac_{args.label_a}={meta_a[f]['wound_pixel_fraction']}  "
              f"wound_frac_{args.label_b}={meta_b[f]['wound_pixel_fraction']}")


if __name__ == "__main__":
    main()
