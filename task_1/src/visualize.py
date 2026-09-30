from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LABEL_HEIGHT = 20
PANEL_GAP = 6


def denormalize_image(image_t):
    img = image_t.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def mask_to_rgb(mask, colors):
    mask = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(colors):
        out[mask == class_id] = rgb
    return out


def overlay_mask(image_rgb, mask_rgb, alpha=0.45):
    blended = image_rgb.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha
    return blended.astype(np.uint8)


def _labeled_panel(array, label):
    panel = Image.fromarray(array)
    canvas = Image.new("RGB", (panel.width, panel.height + LABEL_HEIGHT), (255, 255, 255))
    canvas.paste(panel, (0, LABEL_HEIGHT))
    ImageDraw.Draw(canvas).text((4, 2), label, fill=(0, 0, 0))
    return canvas


def save_sample_panel(image_t, gt_mask, pred_mask, colors, out_path, alpha=0.45):
    """Saves a 3-panel PNG (image | ground truth overlay | prediction overlay)."""
    image_rgb = denormalize_image(image_t)
    gt_overlay = overlay_mask(image_rgb, mask_to_rgb(gt_mask, colors), alpha)
    pred_overlay = overlay_mask(image_rgb, mask_to_rgb(pred_mask, colors), alpha)

    panels = [
        _labeled_panel(image_rgb, "Image"),
        _labeled_panel(gt_overlay, "Ground Truth"),
        _labeled_panel(pred_overlay, "Prediction"),
    ]
    total_w = sum(p.width for p in panels) + PANEL_GAP * (len(panels) - 1)
    total_h = max(p.height for p in panels)
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + PANEL_GAP

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
