#!/usr/bin/env python3
"""Experiment 1 data prep (Chapter 8): zero out everything except skin and
wound in each staging image, using the 3-fold ensemble mask, and save as a
flat classification-style dataset (stage_X/<file>, matching
datasets/wound-stage-classification/'s own layout exactly -- no
images/masks subfolders, no mask files).

Input: ../../../../datasets/wound-staging-masks/stage_X/{images,masks}/
(the ensemble masks from ../../../piid_crops/generate_masks_ensemble.py).
Nothing reads the staging label -- only the mask's per-class split is used.

For each image, pixels whose mask color is NOT skin (#0000FF) or wound
(#FF0000) are zeroed out; everything else (bounding box, resolution) stays
as in the source.

Output:
  datasets/wound-staging-masks-exp-1/
  |-- stage_1/<file>
  |-- stage_2/<file>
  |-- stage_3/<file>
  |-- stage_4/<file>
  `-- metadata.csv   (filename, stage, foreground_fraction)

Usage:
  python generate_dataset.py
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent  # exp/exp1_wound_crop/scripts/
REPO_ROOT = THIS_DIR.parents[3]

STAGE_DIRS = ["stage_1", "stage_2", "stage_3", "stage_4"]
KEEP_RGB = [(0, 0, 255), (255, 0, 0)]  # skin, wound -- "other" (black) is always dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default=None,
                    help="defaults to <repo>/datasets/wound-staging-masks (images+masks pairs)")
    p.add_argument("--output-dir", default=None,
                    help="defaults to <repo>/datasets/wound-staging-masks-exp-1")
    args = p.parse_args()

    source_dir = Path(args.source_dir) if args.source_dir else REPO_ROOT / "datasets" / "wound-staging-masks"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "datasets" / "wound-staging-masks-exp-1"
    keep_rgbs = [np.array(rgb, dtype=np.uint8) for rgb in KEEP_RGB]

    metadata_rows = []

    for stage in STAGE_DIRS:
        img_dir = source_dir / stage / "images"
        mask_dir = source_dir / stage / "masks"
        if not img_dir.is_dir():
            print(f"WARNING: {img_dir} not found, skipping")
            continue

        stage_out = output_dir / stage
        stage_out.mkdir(parents=True, exist_ok=True)

        filenames = sorted(p.name for p in img_dir.iterdir() if p.is_file())
        print(f"{stage}: {len(filenames)} images")

        for i, filename in enumerate(filenames):
            image = np.array(Image.open(img_dir / filename).convert("RGB"))
            mask_path = mask_dir / (Path(filename).stem + ".png")
            mask = np.array(Image.open(mask_path).convert("RGB"))

            foreground = np.zeros(mask.shape[:2], dtype=bool)
            for rgb in keep_rgbs:
                foreground |= np.all(mask == rgb, axis=-1)

            masked_image = image.copy()
            masked_image[~foreground] = 0

            Image.fromarray(masked_image).save(stage_out / filename)

            metadata_rows.append({
                "filename": filename,
                "stage": stage,
                "foreground_fraction": round(float(foreground.mean()), 6),
            })

            if (i + 1) % 100 == 0:
                print(f"  {stage}: {i + 1}/{len(filenames)}", flush=True)

    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "stage", "foreground_fraction"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    n_total = len(metadata_rows)
    print(f"done: {n_total} images, kept classes=skin+wound")
    print(f"wrote {metadata_path}")
    print(f"output dataset at: {output_dir}")


if __name__ == "__main__":
    main()
