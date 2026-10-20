#!/usr/bin/env python3
"""Tissue-statistics feature extraction (Chapter 8, exp-6).

Steps back from a learned CNN tissue encoder (exp-5's weak-gradient
problem) to simple, hand-computed per-class statistics over the 5 tissue
probability maps -- the question this isolates: do the probability maps
carry ANY independent staging signal at all, before trying to learn a
better feature extractor for them.

For each image, for each of the 5 classes (granulation, slough, necrosis,
tendon, bone -- task_2's CLASS_ORDER), computes 4 statistics OVER VALID
WOUND PIXELS ONLY:
  - mean probability
  - std of probability
  - 95th percentile of probability
  - fraction of pixels with probability > 0.5
=> 5 classes x 4 stats = 20-dim feature vector per image.

"Valid wound pixels" = the leakage-free Task 1 ENSEMBLE mask
(datasets/wound-staging-masks/stage_X/masks/, red = wound) -- the exact
same predicted mask exp-3's generate_dataset.py used to gate the tissue
channels in the first place. NOT ground truth: Task 1's mask is itself a
prediction, never reads the staging label or any tissue annotation. If an
image has zero wound pixels (41/1091, per more_experiments/README.md),
all 20 features are 0 -- same deterministic "missing feature" convention
as exp-3/exp-5, not a fabricated value and not mapped to any stage.

Reads the tissue probability values from datasets/wound-staging-masks-exp-3/
(channels 3-7 of each .npy, already uint8-scaled from the [0,1] softmax
outputs exp-3's generate_dataset.py produced -- same provenance, same
leakage-safety, see that dataset's provenance.json). Does not re-run Task 2.

Output: tissue_statistics.csv (filename, stage, n_wound_px, then 20
feature columns named <class>_<stat>) at the exp-6 root, used identically
by both the train and test split of every experiment here -- no
train/test-specific preprocessing.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent  # exp/exp6_tissue_statistics/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]

STAGE_DIRS = ["stage_1", "stage_2", "stage_3", "stage_4"]
CLASS_ORDER = ["granulation", "slough", "necrosis", "tendon", "bone"]
WOUND_RGB = (255, 0, 0)
STATS = ["mean", "std", "p95", "frac_above_0.5"]


def compute_stats(values):
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(values.mean()),
        float(values.std()),
        float(np.percentile(values, 95)),
        float((values > 0.5).mean()),
    ]


def main():
    exp3_dir = REPO_ROOT / "datasets" / "wound-staging-masks-exp-3"
    mask_dir = REPO_ROOT / "datasets" / "wound-staging-masks"

    feature_names = [f"{c}_{s}" for c in CLASS_ORDER for s in STATS]
    rows = []

    for stage in STAGE_DIRS:
        npy_dir = exp3_dir / stage
        wound_mask_dir = mask_dir / stage / "masks"
        if not npy_dir.is_dir():
            print(f"WARNING: {npy_dir} not found, skipping")
            continue

        filenames = sorted(p.name for p in npy_dir.iterdir() if p.suffix == ".npy")
        print(f"{stage}: {len(filenames)} images")

        for fname in filenames:
            arr = np.load(npy_dir / fname)  # (H, W, 8) uint8
            tissue = arr[:, :, 3:].astype(np.float32) / 255.0  # (H, W, 5), [0,1]

            mask_path = wound_mask_dir / (Path(fname).stem + ".png")
            wound_mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            wound_mask = np.all(wound_mask_rgb == np.array(WOUND_RGB, dtype=np.uint8), axis=-1)
            n_wound_px = int(wound_mask.sum())

            feats = []
            for c in range(len(CLASS_ORDER)):
                class_probs = tissue[:, :, c][wound_mask]
                feats.extend(compute_stats(class_probs))

            row = {"filename": fname.replace(".npy", ""), "stage": stage, "n_wound_px": n_wound_px}
            row.update(dict(zip(feature_names, [round(v, 6) for v in feats])))
            rows.append(row)

    out_path = EXP_DIR / "tissue_statistics.csv"
    fieldnames = ["filename", "stage", "n_wound_px"] + feature_names
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_zero = sum(1 for r in rows if r["n_wound_px"] == 0)
    print(f"done: {len(rows)} images, {n_zero} with zero wound pixels (all-zero feature vector)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
