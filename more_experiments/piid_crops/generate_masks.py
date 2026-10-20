#!/usr/bin/env python3
"""Step 2, part 1 (Chapter 8): run the leakage-free Task 1 checkpoint over
the full PIID staging dataset and save predicted wound-region masks,
organized by stage.

Does NOT yet compute bounding boxes or crops -- that needs the
fallback_strategy decision in configs/crop_config.yaml (still an open TODO
for the "no wound detected" case). A dense per-pixel mask has no such
ambiguity: the model always emits one, even if it predicts zero wound
pixels, so this part can run now.

Staging labels are NEVER read by the model or used to influence the
prediction -- the stage_X folder name is only used to copy the image across
and tag metadata afterward.

Writes a new dataset-shaped folder:

  <output-dir>/
  |-- stage_1/{images,masks}/
  |-- stage_2/{images,masks}/
  |-- stage_3/{images,masks}/
  |-- stage_4/{images,masks}/
  |-- color-mappings.json      (copied from wound-region-segmentation)
  `-- metadata.csv             (filename, stage, wound_pixel_fraction, detected)

Usage:
  python generate_masks.py --checkpoint ../task1_retrain/checkpoints/manet__mit_b2__imagenet_leakagefree_fold0.pt
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "task_1"))

from src.dataset import CLASS_ORDER, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.model import build_model  # noqa: E402

STAGE_DIRS = ["stage_1", "stage_2", "stage_3", "stage_4"]


def load_colors(color_map_path):
    with open(color_map_path) as f:
        mapping = json.load(f)
    colors = []
    for name in CLASS_ORDER:
        hexcolor = mapping[name].lstrip("#")
        rgb = tuple(int(hexcolor[i:i + 2], 16) for i in (0, 2, 4))
        colors.append(rgb)
    return colors


def mask_to_rgb(mask, colors):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(colors):
        out[mask == class_id] = rgb
    return out


def preprocess(image, image_size):
    image = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(image, dtype=np.uint8).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def get_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                    help="path to a manet__mit_b2__imagenet_leakagefree_fold*.pt state dict")
    p.add_argument("--source-root", default=None,
                    help="defaults to <repo>/datasets/wound-stage-classification")
    p.add_argument("--output-dir", default=None,
                    help="defaults to <repo>/datasets/wound-staging-masks")
    p.add_argument("--image-size", type=int, default=256,
                    help="must match task1_retrain training resolution")
    p.add_argument("--limit-per-stage", type=int, default=None, help="for a quick sanity run")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    source_root = Path(args.source_root) if args.source_root else REPO_ROOT / "datasets" / "wound-stage-classification"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "datasets" / "wound-staging-masks"
    seg_color_map = REPO_ROOT / "datasets" / "wound-region-segmentation" / "color-mappings.json"
    colors = load_colors(seg_color_map)

    device = get_device(args.device)
    model = build_model("manet", "mit_b2", None, num_classes=len(CLASS_ORDER))
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"loaded checkpoint: {args.checkpoint} on {device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(seg_color_map, output_dir / "color-mappings.json")

    metadata_rows = []
    wound_class_idx = CLASS_ORDER.index("wound")

    for stage in STAGE_DIRS:
        src_dir = source_root / stage
        if not src_dir.is_dir():
            print(f"WARNING: {src_dir} not found, skipping")
            continue
        images_out = output_dir / stage / "images"
        masks_out = output_dir / stage / "masks"
        images_out.mkdir(parents=True, exist_ok=True)
        masks_out.mkdir(parents=True, exist_ok=True)

        filenames = sorted(p.name for p in src_dir.iterdir() if p.is_file())
        if args.limit_per_stage:
            filenames = filenames[:args.limit_per_stage]
        print(f"{stage}: {len(filenames)} images")

        for i, filename in enumerate(filenames):
            src_path = src_dir / filename
            image = Image.open(src_path).convert("RGB")
            orig_size = image.size  # (w, h)

            tensor = preprocess(image, args.image_size).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            wound_fraction = float((pred == wound_class_idx).mean())
            detected = wound_fraction > 0.0

            # Resize the predicted mask back to the original image size
            # (NEAREST to preserve discrete class labels), color-coded the
            # same way as wound-region-segmentation/masks/.
            mask_rgb = mask_to_rgb(pred, colors)
            mask_img = Image.fromarray(mask_rgb).resize(orig_size, Image.NEAREST)
            mask_name = Path(filename).stem + ".png"
            mask_img.save(masks_out / mask_name)

            shutil.copy(src_path, images_out / filename)

            metadata_rows.append({
                "filename": filename,
                "stage": stage,
                "wound_pixel_fraction": round(wound_fraction, 6),
                "detected": detected,
            })

            if (i + 1) % 100 == 0:
                print(f"  {stage}: {i + 1}/{len(filenames)}", flush=True)

    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "stage", "wound_pixel_fraction", "detected"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    n_total = len(metadata_rows)
    n_detected = sum(1 for r in metadata_rows if r["detected"])
    print(f"done: {n_total} images, {n_detected} with a detected wound region ({n_total - n_detected} with none)")
    print(f"wrote {metadata_path}")
    print(f"output dataset at: {output_dir}")


if __name__ == "__main__":
    main()
