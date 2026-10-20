#!/usr/bin/env python3
"""Experiment 2 data prep (Chapter 8), with adaptive wound-ROI dilation.

exp-2 keeps the predicted wound region only (unlike exp-1, which also keeps
skin). Two problems with that:

  1. Even an already-large wound got NO surrounding skin at all -- a hard
     cliff at the target coverage. Real wound photos always show some
     peri-wound skin, so every image now gets a fixed --base-margin-px ring
     of skin around the wound (or around the center seed, if nothing was
     detected) unconditionally, not just the small/undetected ones.
  2. Small or undetected wounds were grown by exact circular distance
     thresholding, which produced a suspiciously clean, perfectly round
     blob for every undetected case -- an artifact a model could learn to
     recognize on its own instead of actual wound appearance. Growth beyond
     the base margin is now perturbed by smooth, low-frequency noise so the
     boundary looks organic (like the others' real wound-boundary
     irregularity) instead of a textbook circle.

The amount of *extra* (beyond-base-margin) growth is still inversely
proportional to how small the original wound region was: already covers
>= --target-fraction after the base margin -> no extra growth; smaller ->
grow (noisy, every direction) until target-fraction is reached or
--max-radius-px is hit (safety cap); nothing detected at all -> seed a
small disk at the image center first, then grow the same way.

Input: datasets/wound-staging-masks/stage_X/{images,masks}/ (the ensemble
masks from ../../../piid_crops/generate_masks_ensemble.py). Output: flat
stage_X/<file>, no masks, matching datasets/wound-stage-classification/'s
own layout (same as exp-1).

Usage:
  python generate_dataset.py
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

THIS_DIR = Path(__file__).resolve().parent  # exp/exp2_wound_dilated/scripts/
REPO_ROOT = THIS_DIR.parents[3]

STAGE_DIRS = ["stage_1", "stage_2", "stage_3", "stage_4"]
WOUND_RGB = (255, 0, 0)


def center_seed(shape, radius_px):
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return dist2 <= radius_px ** 2


def apply_base_margin(mask, base_margin_px):
    """Unconditional uniform skin ring around the mask -- applied to every
    image, not just small/undetected ones, so a large wound still shows
    peri-wound skin instead of a hard cut at its exact boundary."""
    if base_margin_px <= 0:
        return mask
    dist = ndimage.distance_transform_edt(~mask)
    return mask | (dist <= base_margin_px)


def smooth_noise_field(shape, amplitude_px, sigma_px, rng):
    """Low-frequency, zero-centered noise (roughly +/- amplitude_px after
    normalisation) used to perturb the growth boundary so it reads as an
    irregular wound-like shape instead of a perfect circle."""
    if amplitude_px <= 0:
        return np.zeros(shape, dtype=np.float32)
    noise = rng.standard_normal(shape).astype(np.float32)
    noise = ndimage.gaussian_filter(noise, sigma=sigma_px)
    peak = np.abs(noise).max()
    if peak > 1e-6:
        noise = noise / peak * amplitude_px
    return noise


def dilate_to_target(mask, target_fraction, max_radius_px, noise_amplitude_px, noise_sigma_px, rng):
    """Grows `mask` outward until its area reaches target_fraction of the
    image, capped at max_radius_px. The growth boundary is the true
    Euclidean distance to the mask PLUS smooth noise, so it comes out
    irregular rather than a clean circle. Original mask pixels are always
    kept regardless of noise. Returns (grown_mask, threshold_used_px).
    """
    total = mask.size
    mask_px = int(mask.sum())
    target_px = target_fraction * total
    if mask_px >= target_px:
        return mask, 0.0

    dist = ndimage.distance_transform_edt(~mask)
    noise = smooth_noise_field(mask.shape, noise_amplitude_px, noise_sigma_px, rng)
    dist_eff = np.clip(dist + noise, 0, None)

    bg_vals = np.sort(dist_eff[~mask])
    needed = target_px - mask_px
    idx = min(int(np.ceil(needed)), len(bg_vals)) - 1
    idx = max(idx, 0)
    threshold = float(bg_vals[idx])

    if threshold > max_radius_px:
        threshold = float(max_radius_px)

    grown = mask | (dist_eff <= threshold)
    return grown, threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default=None,
                    help="defaults to <repo>/datasets/wound-staging-masks (images+masks pairs)")
    p.add_argument("--output-dir", default=None,
                    help="defaults to <repo>/datasets/wound-staging-masks-exp-2")
    p.add_argument("--base-margin-px", type=int, default=12,
                    help="uniform skin ring applied to EVERY image, regardless of wound size "
                         "(default 12 -- so even large wounds keep visible peri-wound skin)")
    p.add_argument("--target-fraction", type=float, default=0.08,
                    help="minimum wound-ROI area fraction (after the base margin) to grow up to")
    p.add_argument("--max-radius-px", type=int, default=70,
                    help="safety cap on extra growth, so a sliver or zero-detection case "
                         "can't balloon to cover the whole frame")
    p.add_argument("--seed-radius-px", type=int, default=3,
                    help="radius of the center seed disk used when a wound wasn't detected at all")
    p.add_argument("--noise-amplitude-px", type=float, default=14.0,
                    help="smooth-noise amplitude perturbing the extra-growth boundary, so it "
                         "looks organic instead of a perfect circle (0 disables)")
    p.add_argument("--noise-sigma-px", type=float, default=12.0,
                    help="smoothing scale of the noise field (larger = gentler, lower-frequency wiggle)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    source_dir = Path(args.source_dir) if args.source_dir else REPO_ROOT / "datasets" / "wound-staging-masks"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "datasets" / "wound-staging-masks-exp-2"
    rng = np.random.default_rng(args.seed)

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

            wound = np.all(mask == np.array(WOUND_RGB, dtype=np.uint8), axis=-1)
            orig_fraction = float(wound.mean())

            used_center_seed = not wound.any()
            seed_mask = wound if not used_center_seed else center_seed(wound.shape, args.seed_radius_px)

            base = apply_base_margin(seed_mask, args.base_margin_px)
            grown, radius_used = dilate_to_target(
                base, args.target_fraction, args.max_radius_px,
                args.noise_amplitude_px, args.noise_sigma_px, rng,
            )
            grown_fraction = float(grown.mean())

            masked_image = image.copy()
            masked_image[~grown] = 0
            Image.fromarray(masked_image).save(stage_out / filename)

            metadata_rows.append({
                "filename": filename,
                "stage": stage,
                "orig_wound_fraction": round(orig_fraction, 6),
                "dilated_fraction": round(grown_fraction, 6),
                "extra_growth_px": round(radius_used, 2),
                "used_center_seed": used_center_seed,
            })

            if (i + 1) % 100 == 0:
                print(f"  {stage}: {i + 1}/{len(filenames)}", flush=True)

    metadata_path = output_dir / "metadata.csv"
    fieldnames = ["filename", "stage", "orig_wound_fraction", "dilated_fraction",
                  "extra_growth_px", "used_center_seed"]
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    n_total = len(metadata_rows)
    n_extra_grown = sum(1 for r in metadata_rows if r["extra_growth_px"] > 0)
    n_seeded = sum(1 for r in metadata_rows if r["used_center_seed"])
    n_capped = sum(1 for r in metadata_rows if r["extra_growth_px"] >= args.max_radius_px)
    print(f"done: {n_total} images, base margin {args.base_margin_px}px on all, "
          f"{n_extra_grown} got extra noisy growth, {n_seeded} used center-seed fallback, "
          f"{n_capped} hit the max-radius cap without reaching target")
    print(f"wrote {metadata_path}")
    print(f"output dataset at: {output_dir}")


if __name__ == "__main__":
    main()
