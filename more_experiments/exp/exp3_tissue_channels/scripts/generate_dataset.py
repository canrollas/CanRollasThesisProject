#!/usr/bin/env python3
"""Experiment 3 data prep (Chapter 8): RGB + 5 tissue-probability channels.

Unlike exp-1/exp-2, the RGB part is NEVER cropped or masked -- every
sample is the full original PIID photo, 3 channels, untouched. Channels
4-8 add per-pixel tissue-class probabilities (granulation, slough,
necrosis, tendon, bone -- task_2/src/dataset.py's CLASS_ORDER, in that
order) from Task 2's segmentation model, so the final array is 8-channel,
same height/width as the source photo.

Two-stage pipeline per image, wound localisation first, THEN tissue:
  1. Find the wound using the leakage-free Task 1 ensemble mask (already
     computed in datasets/wound-staging-masks/stage_X/masks/ -- see
     ../../../piid_crops/generate_masks_ensemble.py). Nothing here re-runs
     Task 1; this only reads its output.
  2. If no wound pixels at all: channels 4-8 are all zero for the whole
     image (the "default empty channels" case) -- Task 2 never runs.
  3. If a wound was found: crop the image to the wound's bounding box (+
     padding), resize to Task 2's own training resolution (512x512, not
     the low-res full frame) and run Task 2 there -- matching the
     resolution regime Task 2 was actually trained on, since its own
     tissue_images are already close-up wound crops, not full photos.
     Average softmax probabilities across every provided Task 2 fold
     checkpoint (ensemble, same approach as Task 1's generate_masks_ensemble.py),
     resize the per-class probability maps back down to the crop's pixel
     footprint, paste into a full-image-sized zero canvas at the bbox
     location, then zero out every pixel outside the actual wound mask
     (not just outside the bbox) -- tissue predictions only mean something
     inside the wound bed.

Task 2 checkpoints: use
task_2/results/task_2_outputs/checkpoints/wound_inner_fold{1,2,3}/MAnet_mit_b2_imagenet.pth
directly -- these are the real, current, properly-evaluated MAnet+MiT-B2
weights (see more_experiments/README.md "Checkpoint gaps"), just saved
under an older naming convention. Ensembling all 3 is recommended,
consistent with how Task 1's generate_masks_ensemble.py does it.

Leakage check (tissue model vs. staging evaluation images): Task 2 was
trained on datasets/wound-tissue-segmentation/tissue_images/ (612 images,
488 unique clinical-photo case-uuids). Verified directly (see
more_experiments/manifests/piid_overlap_report.json ->
task2_piid_overlap_check, and more_experiments/README.md section 1) that
this source has ZERO filename/uuid overlap with the PIID staging dataset
(datasets/wound-stage-classification/) these tissue predictions are being
generated for -- no PIID image or patient was ever seen during Task 2
training, in any fold. Because of that, there is no "which Task 2 fold is
this staging image's fold" question to get wrong (unlike leaking Task 1 on
its own PIID subset, which is exactly what task1_retrain/ exists to avoid)
-- every staging image is equally out-of-distribution for all 3 Task 2
folds, so ensembling all 3 for every staging image is leakage-safe by
construction, not something that needs a per-image fold lookup.

Output: one file per image, same stage_X/<file> layout as exp-1/exp-2 but
saved as .npy (uint8, H x W x 8) since 8 channels don't fit a JPEG/PNG --
<file> keeps the original stem with a .npy extension. A provenance.json
at the dataset root records checkpoint paths, class order, and this
leakage-check result, so the generated predictions are auditable without
re-reading this docstring.

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

THIS_DIR = Path(__file__).resolve().parent  # exp/exp3_tissue_channels/scripts/
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


def get_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    model = build_model("manet", "mit_b2", None, num_classes=len(CLASS_ORDER))
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
                    help="defaults to <repo>/datasets/wound-staging-masks-exp-3")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    for ckpt in args.checkpoints:
        if not Path(ckpt).is_file():
            raise SystemExit(f"checkpoint not found: {ckpt}")

    mask_source_dir = Path(args.mask_source_dir) if args.mask_source_dir else REPO_ROOT / "datasets" / "wound-staging-masks"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "datasets" / "wound-staging-masks-exp-3"

    device = get_device(args.device)
    models = [load_model(c, device) for c in args.checkpoints]
    print(f"loaded {len(models)} Task 2 checkpoint(s) on {device}: {[Path(c).name for c in args.checkpoints]}")
    print("classes (channel order 4-8):", CLASS_ORDER)

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator_script": "more_experiments/exp/exp3_tissue_channels/scripts/generate_dataset.py",
        "channel_order": ["R", "G", "B"] + CLASS_ORDER,
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
            "for every Task 2 fold)."
        ),
        "wound_localization_source": "datasets/wound-staging-masks (leakage-free Task 1 ensemble, see piid_crops/generate_masks_ensemble.py)",
        "wound_localization_architecture": "manet + mit_b2, trained on 3,248 leakage-free Task 1 images (more_experiments/task1_retrain)",
        "task2_inference_resolution_px": TASK2_IMAGE_SIZE,
        "bbox_padding_px": BBOX_PADDING_PX,
        "no_wound_detected_behavior": (
            "RGB channels keep the real, unmodified original photo (never replaced with a "
            "synthetic/black image). Tissue channels (3-7) are all zero -- this is a "
            "missing-feature signal, not a label: the staging label itself is untouched, "
            "and a failed detection is never mapped to or conflated with Stage 1."
        ),
    }
    with open(output_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"wrote {output_dir / 'provenance.json'}")

    metadata_rows = []
    num_classes = len(CLASS_ORDER)

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

            tissue_channels = np.zeros((h, w, num_classes), dtype=np.uint8)
            wound_detected = bool(wound.any())

            if wound_detected:
                x0, y0, x1, y1 = wound_bbox(wound, BBOX_PADDING_PX, (h, w))
                crop_rgb = image_rgb[y0:y1, x0:x1]
                probs = predict_tissue_probs(models, crop_rgb, device)  # (C, crop_h, crop_w)

                crop_wound = wound[y0:y1, x0:x1]
                probs = probs * crop_wound[None, :, :]  # zero outside the actual wound pixels, not just the bbox
                probs_u8 = np.clip(probs * 255.0, 0, 255).astype(np.uint8)  # (C, crop_h, crop_w)
                tissue_channels[y0:y1, x0:x1, :] = probs_u8.transpose(1, 2, 0)

            combined = np.concatenate([image_rgb, tissue_channels], axis=-1)  # (H, W, 8) uint8
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
    print(f"done: {n_total} images, {n_detected} got real tissue channels, "
          f"{n_total - n_detected} got default-empty channels (no wound found)")
    print(f"wrote {metadata_path}")
    print(f"output dataset at: {output_dir}")
    print("channel order: 0-2=RGB, 3=granulation, 4=slough, 5=necrosis, 6=tendon, 7=bone")


if __name__ == "__main__":
    main()
