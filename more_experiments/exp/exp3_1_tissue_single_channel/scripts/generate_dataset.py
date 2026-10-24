#!/usr/bin/env python3
"""Experiment 3.1 data prep (Chapter 8): RGB + 1 severity-banded tissue channel.

Same two-stage pipeline as exp-3 (wound localisation via Task 1, then Task
2 tissue-class softmax probabilities inside the wound bbox), but the 5
per-class probability channels are collapsed into a SINGLE channel before
saving, via encode_severity_channel() below:

  0       -- no wound detected (channels 4-8 in exp-3's convention; here
             just channel 3) -- a missing-feature signal, not a label.
  1-255   -- split into 5 equal ~51-wide bands, one per Task 2 class, in
             CLASS_ORDER's severity order (granulation < slough < necrosis
             < tendon < bone, task_2/src/dataset.py's own class order,
             already used as an increasing-severity order in exp-3's
             docstrings). band = argmax class's index; position within the
             band = that class's own softmax confidence (0 = bottom of the
             band, 1 = top). So the channel jointly encodes WHICH tissue
             dominates a pixel and HOW CONFIDENT that call is, instead of
             5 separate probability maps.

Everything else (Task 1 mask reuse, Task 2 ensemble, leakage check,
bbox/crop/resize geometry, "no wound -> real RGB + zero tissue channel"
convention) is identical to exp-3's scripts/generate_dataset.py -- see that
file's docstring for the full provenance discussion. This script only
changes what happens to the per-class probability map AFTER Task 2
inference, not how Task 1/Task 2 are run.

Output: one file per image, same stage_X/<file> layout as exp-3 but
4-channel (H x W x 4 uint8: RGB + 1 severity channel) instead of 8-channel.
A provenance.json at the dataset root records checkpoint paths, the
severity-band assignment, and the leakage-check result.

Usage:
  python generate_dataset.py --checkpoints \
      ../../../../task_2/results/task_2_outputs/checkpoints/wound_inner_fold1/MAnet_mit_b2_imagenet.pth \
      ../../../../task_2/results/task_2_outputs/checkpoints/wound_inner_fold2/MAnet_mit_b2_imagenet.pth \
      ../../../../task_2/results/task_2_outputs/checkpoints/wound_inner_fold3/MAnet_mit_b2_imagenet.pth
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent  # exp/exp3_1_tissue_single_channel/scripts/
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_2"))

from src.model import build_model  # noqa: E402
from src.dataset import CLASS_ORDER  # noqa: E402  (granulation, slough, necrosis, tendon, bone)

STAGE_DIRS = ["stage_1", "stage_2", "stage_3", "stage_4"]
WOUND_RGB = (255, 0, 0)
TASK2_IMAGE_SIZE = 512  # task_2/configs/base.yaml's training resolution
BBOX_PADDING_PX = 16

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_CLASSES = len(CLASS_ORDER)  # 5
BAND_WIDTH = 255 // NUM_CLASSES  # 51 -- bands live in [1, 255], 0 reserved for "no wound"


def encode_severity_channel(probs):
    """probs: (C, H, W) softmax probabilities (mutually exclusive classes,
    CLASS_ORDER's severity order). Returns (H, W) uint8: band(argmax class)
    + confidence-scaled position within that band, in [1, 255]. Caller is
    responsible for zeroing pixels outside the wound (that's the "no
    tissue signal" case, kept at 0, never produced by this function)."""
    argmax_class = probs.argmax(axis=0)  # (H, W), int in [0, NUM_CLASSES-1]
    confidence = np.take_along_axis(probs, argmax_class[None, :, :], axis=0)[0]  # (H, W), in [0, 1]
    band_start = 1 + argmax_class * BAND_WIDTH
    offset_in_band = np.clip(np.round(confidence * (BAND_WIDTH - 1)), 0, BAND_WIDTH - 1)
    return (band_start + offset_in_band).astype(np.uint8)


def get_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    model = build_model("manet", "mit_b2", None, num_classes=NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def wound_bbox(wound_mask, padding_px, image_shape):
    ys, xs = np.where(wound_mask)
    h, w = image_shape
    x0 = max(int(xs.min()) - padding_px, 0)
    y0 = max(int(ys.min()) - padding_px, 0)
    x1 = min(int(xs.max()) + 1 + padding_px, w)
    y1 = min(int(ys.max()) + 1 + padding_px, h)
    return x0, y0, x1, y1


def preprocess_crop(crop_rgb):
    image = Image.fromarray(crop_rgb).resize((TASK2_IMAGE_SIZE, TASK2_IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.uint8).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def predict_tissue_probs(models, crop_rgb, device):
    """Returns (num_classes, crop_h, crop_w) averaged softmax probabilities,
    resized back to the crop's own pixel size."""
    crop_h, crop_w = crop_rgb.shape[:2]
    tensor = preprocess_crop(crop_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        probs_sum = None
        for model in models:
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        mean_probs = probs_sum / len(models)  # (1, C, 512, 512)
        resized = F.interpolate(mean_probs, size=(crop_h, crop_w), mode="bilinear", align_corners=False)
    return resized.squeeze(0).cpu().numpy()  # (C, crop_h, crop_w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                    help="1+ Task 2 (manet+mit_b2) fold checkpoint paths to ensemble")
    p.add_argument("--mask-source-dir", default=None,
                    help="defaults to <repo>/datasets/wound-staging-masks (Task 1 ensemble masks)")
    p.add_argument("--output-dir", default=None,
                    help="defaults to <repo>/datasets/wound-staging-masks-exp-3-1")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    for ckpt in args.checkpoints:
        if not Path(ckpt).is_file():
            raise SystemExit(f"checkpoint not found: {ckpt}")

    mask_source_dir = Path(args.mask_source_dir) if args.mask_source_dir else REPO_ROOT / "datasets" / "wound-staging-masks"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "datasets" / "wound-staging-masks-exp-3-1"

    device = get_device(args.device)
    models = [load_model(c, device) for c in args.checkpoints]
    print(f"loaded {len(models)} Task 2 checkpoint(s) on {device}: {[Path(c).name for c in args.checkpoints]}")
    print("severity bands (low->high):", CLASS_ORDER)

    bands = {cls: (1 + i * BAND_WIDTH, min(1 + (i + 1) * BAND_WIDTH - 1, 255)) for i, cls in enumerate(CLASS_ORDER)}

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator_script": "more_experiments/exp/exp3_1_tissue_single_channel/scripts/generate_dataset.py",
        "channel_order": ["R", "G", "B", "tissue_severity"],
        "severity_encoding": {
            "description": (
                "0 = no wound detected. 1-255 = band(argmax class) + "
                "confidence-scaled offset within that band. Band order is "
                "CLASS_ORDER's severity order (low to high)."
            ),
            "class_order_low_to_high_severity": CLASS_ORDER,
            "band_width": BAND_WIDTH,
            "band_ranges": {cls: list(rng) for cls, rng in bands.items()},
            "formula": "value = 1 + argmax_class_idx * band_width + round(confidence * (band_width - 1))",
        },
        "task2_architecture": "manet + mit_b2",
        "task2_checkpoints": [str(Path(c).resolve().relative_to(REPO_ROOT)) for c in args.checkpoints],
        "task2_ensemble_method": "mean of per-fold softmax probabilities",
        "task2_training_data": "datasets/wound-tissue-segmentation/tissue_images (612 images, 488 unique clinical-photo case-uuids)",
        "task2_activation": "softmax over 5 classes (mutually exclusive)",
        "leakage_check": (
            "Verified zero filename/uuid overlap between Task 2's training source "
            "(wound-tissue-segmentation) and the PIID staging dataset these predictions "
            "are generated for (wound-stage-classification) -- see "
            "more_experiments/manifests/piid_overlap_report.json 'task2_piid_overlap_check' "
            "and more_experiments/README.md section 1. No staging image or patient was in "
            "Task 2's training data in any fold, so ensembling all 3 Task 2 folds per "
            "staging image is leakage-safe (every staging image is equally out-of-sample "
            "for every Task 2 fold). Identical setup to exp-3, re-verified here since this "
            "is a separate dataset build."
        ),
        "wound_localization_source": "datasets/wound-staging-masks (leakage-free Task 1 ensemble, see piid_crops/generate_masks_ensemble.py)",
        "wound_localization_architecture": "manet + mit_b2, trained on 3,248 leakage-free Task 1 images (more_experiments/task1_retrain)",
        "task2_inference_resolution_px": TASK2_IMAGE_SIZE,
        "bbox_padding_px": BBOX_PADDING_PX,
        "no_wound_detected_behavior": (
            "RGB channels keep the real, unmodified original photo (never replaced with a "
            "synthetic/black image). The severity channel is 0 for the whole image -- this "
            "is a missing-feature signal, not a label: the staging label itself is untouched, "
            "and a failed detection is never mapped to or conflated with the lowest-severity band."
        ),
    }
    with open(output_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"wrote {output_dir / 'provenance.json'}")

    metadata_rows = []

    for stage in STAGE_DIRS:
        img_dir = mask_source_dir / stage / "images"
        wound_mask_dir = mask_source_dir / stage / "masks"
        if not img_dir.is_dir():
            print(f"WARNING: {img_dir} not found, skipping")
            continue

        stage_out = output_dir / stage
        stage_out.mkdir(parents=True, exist_ok=True)

        filenames = sorted(p.name for p in img_dir.iterdir() if p.is_file())
        print(f"{stage}: {len(filenames)} images")

        for i, filename in enumerate(filenames):
            image_rgb = np.array(Image.open(img_dir / filename).convert("RGB"))
            h, w = image_rgb.shape[:2]

            mask_path = wound_mask_dir / (Path(filename).stem + ".png")
            task1_mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            wound = np.all(task1_mask_rgb == np.array(WOUND_RGB, dtype=np.uint8), axis=-1)

            severity_channel = np.zeros((h, w), dtype=np.uint8)
            wound_detected = bool(wound.any())

            if wound_detected:
                x0, y0, x1, y1 = wound_bbox(wound, BBOX_PADDING_PX, (h, w))
                crop_rgb = image_rgb[y0:y1, x0:x1]
                probs = predict_tissue_probs(models, crop_rgb, device)  # (C, crop_h, crop_w)

                crop_severity = encode_severity_channel(probs)  # (crop_h, crop_w) uint8, in [1, 255]
                crop_wound = wound[y0:y1, x0:x1]
                crop_severity = crop_severity * crop_wound  # zero outside the actual wound pixels, not just the bbox
                severity_channel[y0:y1, x0:x1] = crop_severity

            combined = np.concatenate([image_rgb, severity_channel[:, :, None]], axis=-1)  # (H, W, 4) uint8
            out_path = stage_out / (Path(filename).stem + ".npy")
            np.save(out_path, combined)

            metadata_rows.append({
                "filename": filename,
                "stage": stage,
                "wound_detected": wound_detected,
                "npy_file": out_path.name,
            })

            if (i + 1) % 100 == 0:
                print(f"  {stage}: {i + 1}/{len(filenames)}", flush=True)

    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "stage", "wound_detected", "npy_file"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    n_total = len(metadata_rows)
    n_detected = sum(1 for r in metadata_rows if r["wound_detected"])
    print(f"done: {n_total} images, {n_detected} got a real severity channel, "
          f"{n_total - n_detected} got a default-empty channel (no wound found)")
    print(f"wrote {metadata_path}")
    print(f"output dataset at: {output_dir}")
    print(f"channel order: 0-2=RGB, 3=severity (0=no wound, bands: {bands})")


if __name__ == "__main__":
    main()
