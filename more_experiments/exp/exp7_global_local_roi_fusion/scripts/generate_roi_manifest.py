#!/usr/bin/env python3
"""Experiment 7, Step 2 (Chapter 8): wound ROI bounding-box manifest.

Reuses the ALREADY-GENERATED leakage-free Task 1 ensemble masks
(datasets/wound-staging-masks/stage_X/{images,masks}/, metadata.csv) from
more_experiments/piid_crops/generate_masks_ensemble.py -- that script ran
all 3 leakage-free MAnet+MiT-B2 folds (more_experiments/task1_retrain/
checkpoints/, verified PIID-excluded) over all 1,091 PIID images and
soft-voted the per-pixel class probabilities. This script does NOT re-run
Task 1 inference (same checkpoints + same images -> deterministic eval-mode
forward pass would reproduce the identical masks); it only derives a
bounding box from each mask.

Fixed, documented rules (no staging labels or ground-truth masks used
anywhere here -- only the predicted wound-class mask):

  1. Wound pixels = mask pixels matching color-mappings.json's "wound"
     RGB (#FF0000), read from the per-image mask PNG (saved at the
     ORIGINAL image resolution by generate_masks_ensemble.py).
  2. Component selection: scipy.ndimage.label (default 4-connectivity) ->
     keep the SINGLE LARGEST connected component only. No size threshold,
     no merging of multiple components -- one fixed rule, always.
  3. Bounding box = the selected component's pixel extent, in ORIGINAL
     image coordinates.
  4. Convert to the 224x224 staging coordinate system with INDEPENDENT
     x/y scale factors (orig_w -> 224, orig_h -> 224) -- the same
     non-uniform resize task_3/src/dataset.py's StageDataset already
     applies to the RGB input, so ROI boxes and RGB pixels stay aligned.
  5. Add a fixed 8px margin in 224-space (clamped to [0,224]).
  6. detected=False (metadata.csv, 41/1091 images with zero wound pixels
     at all) -> fallback box = the FULL 224x224 image. Never excluded.

Fold ID: task_3/src/splits.py build_folds(wound-stage-classification, ...)
(same seed=42 stratified 3-fold split as every other more_experiments
experiment), matched to this dataset's images by filename stem.

Output: ../results/roi_manifest.csv with columns
  filename, fold, x1, y1, x2, y2, detected, wound_area_pct
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

THIS_DIR = Path(__file__).resolve().parent  # exp/exp7_global_local_roi_fusion/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.splits import build_folds  # noqa: E402

STAGE_DIRS = ["stage_1", "stage_2", "stage_3", "stage_4"]
WOUND_RGB = (255, 0, 0)
MARGIN_PX_224 = 8
TARGET_SIZE = 224

MASKS_ROOT = REPO_ROOT / "datasets" / "wound-staging-masks"
RGB_DATA_ROOT = REPO_ROOT / "datasets" / "wound-stage-classification"


def largest_component_bbox(wound_mask):
    """wound_mask: (H, W) bool. Returns (y1, x1, y2, x2) pixel extent of the
    largest connected component, or None if wound_mask has no True pixels."""
    labeled, n_components = ndimage.label(wound_mask)
    if n_components == 0:
        return None
    sizes = ndimage.sum_labels(wound_mask, labeled, index=range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labeled == largest_label)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1, int(sizes[largest_label - 1])


def to_224_coords(y1, x1, y2, x2, orig_h, orig_w):
    sx = TARGET_SIZE / orig_w
    sy = TARGET_SIZE / orig_h
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def apply_margin(x1, y1, x2, y2, margin=MARGIN_PX_224, size=TARGET_SIZE):
    x1 = max(0.0, x1 - margin)
    y1 = max(0.0, y1 - margin)
    x2 = min(float(size), x2 + margin)
    y2 = min(float(size), y2 + margin)
    return x1, y1, x2, y2


def main():
    base_cfg_data_root = RGB_DATA_ROOT
    folds = build_folds(base_cfg_data_root, num_folds=3, val_fraction=0.1, seed=42)
    fold_of_stem = {}
    for fold_idx, fold in enumerate(folds):
        for split in ("train", "val", "test"):
            for rel_path, _ in fold[split]:
                stem = Path(rel_path).stem
                if split == "test":  # each image is in exactly one fold's TEST split
                    fold_of_stem[stem] = fold_idx

    rows = []
    n_detected = n_fallback = 0
    for stage in STAGE_DIRS:
        img_dir = MASKS_ROOT / stage / "images"
        mask_dir = MASKS_ROOT / stage / "masks"
        filenames = sorted(p.name for p in img_dir.iterdir() if p.is_file())
        print(f"{stage}: {len(filenames)} images")

        for filename in filenames:
            stem = Path(filename).stem
            image = Image.open(img_dir / filename)
            orig_w, orig_h = image.size
            mask = np.array(Image.open(mask_dir / (stem + ".png")).convert("RGB"))
            wound = np.all(mask == np.array(WOUND_RGB, dtype=np.uint8), axis=-1)

            comp = largest_component_bbox(wound)
            if comp is None:
                x1, y1, x2, y2 = 0.0, 0.0, float(TARGET_SIZE), float(TARGET_SIZE)
                detected = False
                wound_area_pct = 0.0
                n_fallback += 1
            else:
                y1_px, x1_px, y2_px, x2_px, comp_px = comp
                x1, y1, x2, y2 = to_224_coords(y1_px, x1_px, y2_px, x2_px, orig_h, orig_w)
                x1, y1, x2, y2 = apply_margin(x1, y1, x2, y2)
                detected = True
                wound_area_pct = round(100.0 * comp_px / (orig_h * orig_w), 4)
                n_detected += 1

            rows.append({
                "filename": stem, "fold": fold_of_stem[stem],
                "x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2),
                "detected": detected, "wound_area_pct": wound_area_pct,
            })

    out_path = EXP_DIR / "results" / "roi_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "fold", "x1", "y1", "x2", "y2", "detected", "wound_area_pct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} images: {n_detected} detected, {n_fallback} fallback (full-image box)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
