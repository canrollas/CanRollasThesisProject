#!/usr/bin/env python3
"""Experiment 14 data prep (Chapter 8): RGB + 1 monocular relative-depth channel.

Unlike exp-1/2/3/3.1 (which run Task 1 and/or Task 2 inference), this
experiment needs no model inference at all: the depth maps already exist
at `datasets/wound-stage-depth-anything-v2-small/` (Depth Anything V2
Small, generated earlier for WP5's rule-based Stage-3/4 check -- see
`more_experiments/experiment_extended/README.md`), and every one of the
1,091 PIID images in `datasets/wound-stage-classification/` is already
224x224, pixel-for-pixel the same size as its depth map (verified
directly over all 1,091 pairs before writing this script, not assumed).
So this script only has to: load the RGB jpeg, load the matching depth
.npy, min-max normalize the depth map TO [0, 255] PER IMAGE, and stack.

Why per-image min-max (not a global constant, not z-score against a
corpus-wide mean/std): Depth Anything's output is a *relative* inverse
depth map -- `provenance.json`'s own "interpretation" field calls it an
"image-derived geometric prior; no metric or measured wound depth". Each
image's absolute scale/offset is arbitrary (that's what "relative" means
here), only the within-image ordering (what's nearer/farther than what,
inside this one photo) is meaningful. Per-image min-max preserves that
ordering and maps it into the same [0, 255] representation exp-3.1 used
for its tissue-severity channel, so dataset.py/model.py need no changes
beyond renaming. A flat depth map (min==max, no finite-geometry signal --
checked directly, happens for 0/1091 images at generation time but
handled defensively) is encoded as all-128 (mid-gray: "no depth contrast
detected", not 0 -- 0 would look identical to "very near" under min-max
scaling, i.e. a real value, not a missing-feature sentinel like exp-1/2/
3/3.1's "no wound detected" zero convention. There is no missing-feature
case here: Depth Anything always produces an output, unlike Task 1/2
detection, which can fail).

No wound-localisation or tissue dependency whatsoever -- this experiment
tests whether raw monocular geometry alone (not masked, not restricted to
a predicted wound box) helps staging, independent of exp-1/2/3's leakage-
free Task 1 model or exp-3/3.1's Task 2 model.

Output: one file per image, same stage_X/<file> layout as the source
RGB/depth dirs, 4-channel .npy (H x W x 4 uint8: RGB + depth). A
provenance.json at the dataset root records the source dirs, the hash of
the depth generator's own provenance (so this dataset is traceable back
to its exact Depth-Anything-V2-Small run), and the normalization rule.

Usage:
  python generate_dataset.py
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent  # exp/exp14_depth_channel/scripts/
REPO_ROOT = THIS_DIR.parents[3]

STAGE_DIRS = ["stage_1", "stage_2", "stage_3", "stage_4"]


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_depth_channel(depth):
    """depth: (H, W) float32 relative inverse depth. Returns (H, W) uint8,
    per-image min-max normalized to [0, 255]. Flat maps (max == min) fall
    back to a constant 128 -- "no contrast", not "nearest" or "farthest"."""
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-8:
        return np.full(depth.shape, 128, dtype=np.uint8)
    normalized = (depth - d_min) / (d_max - d_min)
    return np.round(normalized * 255.0).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-source-dir", default=None,
                    help="defaults to <repo>/datasets/wound-stage-classification")
    p.add_argument("--depth-source-dir", default=None,
                    help="defaults to <repo>/datasets/wound-stage-depth-anything-v2-small")
    p.add_argument("--output-dir", default=None,
                    help="defaults to <repo>/datasets/wound-staging-rgbd-exp-14")
    args = p.parse_args()

    rgb_source_dir = Path(args.rgb_source_dir) if args.rgb_source_dir else REPO_ROOT / "datasets" / "wound-stage-classification"
    depth_source_dir = Path(args.depth_source_dir) if args.depth_source_dir else REPO_ROOT / "datasets" / "wound-stage-depth-anything-v2-small"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "datasets" / "wound-staging-rgbd-exp-14"

    if not rgb_source_dir.is_dir():
        raise SystemExit(f"RGB source not found: {rgb_source_dir}")
    if not depth_source_dir.is_dir():
        raise SystemExit(f"depth source not found: {depth_source_dir}")

    depth_provenance_path = depth_source_dir / "provenance.json"
    depth_provenance_sha256 = file_sha256(depth_provenance_path) if depth_provenance_path.is_file() else None

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator_script": "more_experiments/exp/exp14_depth_channel/scripts/generate_dataset.py",
        "channel_order": ["R", "G", "B", "relative_depth"],
        "depth_encoding": {
            "description": (
                "Per-image min-max normalization of Depth Anything V2 Small's raw "
                "float32 relative inverse depth to [0, 255] uint8. Flat maps "
                "(max - min < 1e-8) encoded as constant 128."
            ),
            "formula": "round(255 * (depth - depth.min()) / (depth.max() - depth.min()))",
            "flat_map_fallback": 128,
        },
        "rgb_source_dir": str(rgb_source_dir.resolve().relative_to(REPO_ROOT)),
        "depth_source_dir": str(depth_source_dir.resolve().relative_to(REPO_ROOT)),
        "depth_source_provenance_sha256": depth_provenance_sha256,
        "depth_source_model": "depth-anything/Depth-Anything-V2-Small-hf (see depth_source_dir/provenance.json)",
        "depth_interpretation": (
            "Monocular, single-image, scale/shift-ambiguous relative inverse depth -- "
            "an image-derived geometric prior, NOT metric or measured wound depth. "
            "Same interpretation as WP5 (more_experiments/experiment_extended/wp5_depth_stage34/)."
        ),
        "no_masking_or_localization": (
            "Unlike exp-1/2 (Task 1 masking) or exp-3/3.1 (Task 2 tissue probabilities "
            "inside a predicted wound box), this experiment uses the FULL, unmasked RGB "
            "photo and the FULL depth map -- no Task 1 or Task 2 dependency, no leakage "
            "surface from either of those models."
        ),
        "alignment_check": (
            "All 1,091 RGB/depth pairs verified pixel-dimension-identical before this "
            "script was written (no resize step needed or performed)."
        ),
    }
    with open(output_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"wrote {output_dir / 'provenance.json'}")

    metadata_rows = []
    n_flat = 0

    for stage in STAGE_DIRS:
        img_dir = rgb_source_dir / stage
        depth_dir = depth_source_dir / stage
        if not img_dir.is_dir():
            print(f"WARNING: {img_dir} not found, skipping")
            continue

        stage_out = output_dir / stage
        stage_out.mkdir(parents=True, exist_ok=True)

        filenames = sorted(p.name for p in img_dir.iterdir() if p.suffix == ".jpg")
        print(f"{stage}: {len(filenames)} images")

        for i, filename in enumerate(filenames):
            image_rgb = np.array(Image.open(img_dir / filename).convert("RGB"))
            depth_path = depth_dir / (Path(filename).stem + ".npy")
            depth = np.load(depth_path)

            if depth.shape != image_rgb.shape[:2]:
                raise ValueError(
                    f"{filename}: depth shape {depth.shape} != RGB shape {image_rgb.shape[:2]}"
                )

            depth_channel = encode_depth_channel(depth)
            is_flat = bool(float(depth.max()) - float(depth.min()) < 1e-8)
            n_flat += int(is_flat)

            combined = np.concatenate([image_rgb, depth_channel[:, :, None]], axis=-1)  # (H, W, 4) uint8
            out_path = stage_out / (Path(filename).stem + ".npy")
            np.save(out_path, combined)

            metadata_rows.append({
                "filename": filename,
                "stage": stage,
                "depth_flat": is_flat,
                "npy_file": out_path.name,
            })

            if (i + 1) % 100 == 0:
                print(f"  {stage}: {i + 1}/{len(filenames)}", flush=True)

    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "stage", "depth_flat", "npy_file"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    n_total = len(metadata_rows)
    print(f"done: {n_total} images, {n_flat} had a flat depth map (encoded as constant 128)")
    print(f"wrote {metadata_path}")
    print(f"output dataset at: {output_dir}")
    print("channel order: 0-2=RGB, 3=per-image min-max normalized relative depth")


if __name__ == "__main__":
    main()
