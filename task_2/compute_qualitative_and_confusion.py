#!/usr/bin/env python3
"""Re-evaluates already-trained Task 2 checkpoints on their held-out test
folds to produce the three things run_ablation.py does not: a full,
void-aware confusion matrix, per-class tissue-proportion accuracy, and a
curated set of qualitative (image / ground truth / prediction) panels.

Does not retrain anything; only loads each saved checkpoint
(checkpoints/{config_id}_fold{N}.pt, written by run_ablation.py) and runs
inference, using the exact same fold split (same seed) so each checkpoint
sees the same test files it was evaluated on originally. A config with no
checkpoints for any fold is skipped with a warning rather than failing the
whole run, since checkpoints may live on a different machine (e.g. Colab/
Drive) than this script runs on.

Tissue-proportion accuracy is defined exactly as in the thesis (chapter 6,
"Tissue Proportion Accuracy"): for class k and a given image's valid
(non-void) pixels, r_k = N_k / sum_j N_j is the ground-truth proportion and
r_hat_k the same computed from the model's own predicted labels; the metric
reported per class is the mean absolute difference |r_hat_k - r_k|, in
percentage points, averaged over every test image across all folds for that
config.

Qualitative panels: for each of the 5 classes, the test image (across all
folds) with the largest ground-truth pixel count of that class is rendered
as a 3-panel (image / ground truth / prediction) PNG, so each class has one
clear, representative example. The single test image with the worst overall
per-image mIoU, across the whole test set, is also rendered as a failure-case
example.

Writes three outputs incrementally (one config at a time, so a Colab
disconnect only loses the config in progress, not everything before it):
  - results/confusion_matrices.csv         -- summed 5x5 confusion matrix (pixel counts), per config
  - results/tissue_proportion_accuracy.csv -- mean abs. proportion error per class, per config
  - results/qual_panels/{config_id}_{class}_best.png  (x5 per config)
  - results/qual_panels/{config_id}_hardest.png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR))

from src.dataset import TissueDataset, load_color_mapping  # noqa: E402
from src.metrics import compute_confusion, iou_dice_from_confusion  # noqa: E402
from src.model import build_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402
from src.visualize import (  # noqa: E402
    PANEL_GAP, _labeled_panel, denormalize_image, mask_to_rgb, overlay_mask,
)
from PIL import Image  # noqa: E402

CLASS_NAMES = ["granulation", "slough", "necrosis", "tendon", "bone"]
VOID_LABEL = 255

CONF_CSV = "confusion_matrices.csv"
PROP_CSV = "tissue_proportion_accuracy.csv"
QUAL_DIR = "qual_panels"

# The 37-config x 3-fold checkpoints actually sitting on disk are under
# results/task_2_outputs/checkpoints/, using an older naming convention
# (capitalized architecture, 1-indexed fold directories, no config_id
# separator) from before this task's current checkpoint_dir/{config_id}_
# fold{N}.pt convention existed. results/task_2_outputs/results/fold_summary_wound_inner.csv
# was itself computed by evaluating exactly these checkpoints against this
# task's current build_folds() split, so loading them against that same
# split here is consistent with how that CSV was produced -- these are
# the real, current checkpoints, just in the older naming layout.
OUTPUTS_CHECKPOINT_DIR = "results/task_2_outputs/checkpoints"
ARCH_FILENAME = {
    "unet": "Unet", "unetplusplus": "UnetPlusPlus", "manet": "MAnet",
    "deeplabv3plus": "DeepLabV3Plus", "deeplabv3": "DeepLabV3",
    "linknet": "Linknet", "fpn": "FPN", "pspnet": "PSPNet", "pan": "PAN",
}


def outputs_checkpoint_path(task_dir, cfg, fold_idx):
    arch_name = ARCH_FILENAME[cfg["architecture"]]
    filename = f"{arch_name}_{cfg['encoder']}_{cfg['weights']}.pth"
    return task_dir / OUTPUTS_CHECKPOINT_DIR / f"wound_inner_fold{fold_idx + 1}" / filename


def mask_void_for_display(pred, gt, void_label=VOID_LABEL):
    """Task 2's model has no void output (sec:tissue-architecture) and
    predicts one of the 5 real classes everywhere, including outside the
    wound; for display only (not for any metric above), pixels void in
    ground truth are blanked in the prediction too, so the rendered panel
    does not show tissue-class colour over background/periwound pixels
    nothing in this chapter is ever scored against."""
    display = pred.copy()
    display[gt == void_label] = void_label
    return display


def save_panel_void_masked(image_t, gt_mask, pred_mask, colors, out_path, void_label=VOID_LABEL, alpha=0.45):
    """Like src.visualize.save_sample_panel, but the Image panel is masked
    to the ground-truth-valid (non-void) region too, not just Prediction
    (mask_void_for_display above already handles that one). This dataset's
    tissue_images files are RGBA: alpha marks the wound (opaque) versus
    everything outside it (transparent), but the RGB channels underneath
    the transparent region still hold real pixel data (the photo's
    background/periwound surroundings), which .convert("RGB") (used to load
    the image, src/dataset.py) reveals rather than hides. A plain image
    viewer that respects the alpha channel would show only the wound;
    masking here keeps this panel's Image column consistent with that, and
    with the Ground Truth and Prediction columns already being void-aware."""
    void = gt_mask == void_label
    image_rgb = denormalize_image(image_t).copy()
    image_rgb[void] = 255
    pred_display = mask_void_for_display(pred_mask, gt_mask, void_label)
    gt_overlay = overlay_mask(image_rgb, mask_to_rgb(gt_mask, colors), alpha)
    pred_overlay = overlay_mask(image_rgb, mask_to_rgb(pred_display, colors), alpha)
    gt_overlay[void] = 255
    pred_overlay[void] = 255

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


def find_checkpoint(task_dir, checkpoint_dir, config_id, cfg, fold_idx):
    """Current-style checkpoint first (checkpoint_dir/{config_id}_fold{N}.pt);
    falls back to the older results/task_2_outputs/checkpoints/ layout if
    that is not found."""
    current = checkpoint_dir / f"{config_id}_fold{fold_idx}.pt"
    if current.exists():
        return current
    fallback = outputs_checkpoint_path(task_dir, cfg, fold_idx)
    if fallback.exists():
        return fallback
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None, help="defaults to <repo>/datasets/wound-tissue-segmentation")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--configs", default=None,
                    help="comma-separated config_id allowlist (from configs/resolved_grid.json); "
                         "default: every config_id that has at least one checkpoint on disk")
    return p.parse_args()


def load_done_config_ids(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {row["config_id"] for row in csv.DictReader(f)}


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def tissue_proportions(label, num_classes, void_label=VOID_LABEL):
    """label: 2D array of class indices (ints), possibly containing void_label.
    Returns an (num_classes,) array of each class's share of the valid
    (non-void) pixels; an image with zero valid pixels returns all-NaN."""
    valid = label != void_label
    n_valid = valid.sum()
    if n_valid == 0:
        return np.full(num_classes, np.nan)
    counts = np.array([(label[valid] == k).sum() for k in range(num_classes)], dtype=np.float64)
    return counts / n_valid


def discover_checkpoint_configs(task_dir, checkpoint_dir, by_id):
    seen = set()
    if checkpoint_dir.exists():
        for p in checkpoint_dir.glob("*_fold*.pt"):
            if p.name.endswith(".resume.pt"):
                continue
            stem = p.stem  # e.g. "manet__mit_b2__imagenet_fold0"
            config_id = stem.rsplit("_fold", 1)[0]
            if config_id in by_id:
                seen.add(config_id)
    if not seen:
        for config_id, cfg in by_id.items():
            if outputs_checkpoint_path(task_dir, cfg, 0).exists():
                seen.add(config_id)
    return sorted(seen)


def main():
    args = parse_args()
    base_cfg = load_yaml(TASK_DIR / "configs" / "base.yaml")
    grid = json.loads((TASK_DIR / "configs" / "resolved_grid.json").read_text())
    by_id = {c["config_id"]: c for c in grid}
    num_classes = len(base_cfg["classes"])

    repo_root = TASK_DIR.parent
    data_root = Path(args.data_root) if args.data_root else repo_root / "datasets" / "wound-tissue-segmentation"
    colors = load_color_mapping(data_root)

    folds = build_folds(data_root, colors, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    checkpoint_dir = TASK_DIR / base_cfg["checkpoint_dir"]
    results_dir = TASK_DIR / "results"
    batch_size = args.batch if args.batch is not None else base_cfg["batch_size"]
    device = get_device()

    if args.configs is not None:
        config_ids = [c.strip() for c in args.configs.split(",") if c.strip()]
        missing = [c for c in config_ids if c not in by_id]
        if missing:
            raise SystemExit(f"--configs: unknown config_id(s) not in resolved_grid.json: {missing}")
    else:
        config_ids = discover_checkpoint_configs(TASK_DIR, checkpoint_dir, by_id)
        if not config_ids:
            raise SystemExit(f"no checkpoints found under {checkpoint_dir} or "
                              f"{TASK_DIR / OUTPUTS_CHECKPOINT_DIR} -- pass --configs explicitly, "
                              f"or place checkpoints in one of those locations first")

    conf_path = results_dir / CONF_CSV
    prop_path = results_dir / PROP_CSV
    qual_dir = results_dir / QUAL_DIR
    done = load_done_config_ids(conf_path)

    print(f"device={device} configs={config_ids}", flush=True)

    for config_id in config_ids:
        if config_id in done:
            print(f"skip (already done): {config_id}")
            continue

        cfg = by_id[config_id]
        conf_total = np.zeros((num_classes, num_classes), dtype=np.int64)
        abs_prop_errors = [[] for _ in range(num_classes)]  # per-class list of per-image abs errors (fraction, not %)

        best_per_class = [None] * num_classes  # (true_area, image_t, gt, pred, filename)
        worst_overall = None  # (miou, image_t, gt, pred, filename)

        any_checkpoint = False
        for fold_idx, fold in enumerate(folds):
            ckpt_path = find_checkpoint(TASK_DIR, checkpoint_dir, config_id, cfg, fold_idx)
            if ckpt_path is None:
                print(f"MISSING checkpoint: {config_id} fold{fold_idx} "
                      f"(checked {checkpoint_dir} and {TASK_DIR / OUTPUTS_CHECKPOINT_DIR})", file=sys.stderr)
                continue
            any_checkpoint = True

            test_ds = TissueDataset(fold["test"], data_root, base_cfg["image_size"], colors, transform=None)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=args.workers)

            model = build_model(cfg["architecture"], cfg["encoder"], None, num_classes=num_classes)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()

            idx_in_fold = 0
            with torch.no_grad():
                for images, targets in test_loader:
                    images_dev = images.to(device)
                    targets_dev = targets.to(device)
                    logits = model(images_dev)
                    conf_total += compute_confusion(logits, targets_dev, num_classes, ignore_index=VOID_LABEL).cpu().numpy()

                    preds = logits.argmax(dim=1).cpu().numpy()
                    gts = targets.numpy()
                    for b in range(images.shape[0]):
                        filename = fold["test"][idx_in_fold]
                        idx_in_fold += 1

                        gt_np, pred_np = gts[b], preds[b]
                        r = tissue_proportions(gt_np, num_classes)
                        r_hat = tissue_proportions(pred_np, num_classes)
                        if np.isnan(r).any():
                            continue
                        for k in range(num_classes):
                            abs_prop_errors[k].append(abs(r_hat[k] - r[k]))

                        valid = gt_np != VOID_LABEL
                        n_valid = int(valid.sum())
                        if n_valid == 0:
                            continue
                        correct = int((pred_np[valid] == gt_np[valid]).sum())
                        pixel_acc = correct / n_valid
                        if worst_overall is None or pixel_acc < worst_overall[0]:
                            worst_overall = (pixel_acc, images[b].clone(), gt_np.copy(), pred_np.copy(), filename)

                        for k in range(num_classes):
                            area_k = int((gt_np == k).sum())
                            if area_k == 0:
                                continue
                            if best_per_class[k] is None or area_k > best_per_class[k][0]:
                                best_per_class[k] = (area_k, images[b].clone(), gt_np.copy(), pred_np.copy(), filename)

        if not any_checkpoint:
            print(f"skip (no checkpoints for any fold): {config_id}")
            continue

        conf_row = {"config_id": config_id}
        for i, true_name in enumerate(CLASS_NAMES):
            for j, pred_name in enumerate(CLASS_NAMES):
                conf_row[f"true_{true_name}_pred_{pred_name}"] = int(conf_total[i, j])
        append_row(conf_path, conf_row)

        prop_row = {"config_id": config_id}
        for k, name in enumerate(CLASS_NAMES):
            errs = np.array(abs_prop_errors[k])
            prop_row[f"delta_{name}_pct"] = float(errs.mean() * 100) if len(errs) else float("nan")
            prop_row[f"delta_{name}_n_images"] = int(len(errs))
        append_row(prop_path, prop_row)

        for k, name in enumerate(CLASS_NAMES):
            entry = best_per_class[k]
            if entry is None:
                print(f"[{config_id}] no test image contains class {name}; skipping its qual panel")
                continue
            _, image_t, gt_np, pred_np, filename = entry
            out_path = qual_dir / f"{config_id}_{name}_best.png"
            save_panel_void_masked(image_t, gt_np, pred_np, colors, out_path)
            print(f"[{config_id}] wrote {out_path} (from {filename})")

        if worst_overall is not None:
            pixel_acc, image_t, gt_np, pred_np, filename = worst_overall
            out_path = qual_dir / f"{config_id}_hardest.png"
            save_panel_void_masked(image_t, gt_np, pred_np, colors, out_path)
            print(f"[{config_id}] wrote {out_path} (from {filename}, pixel_acc={pixel_acc:.4f})")

        iou, dice = iou_dice_from_confusion(conf_total)
        print(f"[{config_id}] recomputed from confusion matrix: "
              f"mIoU={iou.mean():.4f} mDice={dice.mean():.4f} "
              f"per-class IoU={np.round(iou, 4).tolist()}", flush=True)
        print(f"-- wrote rows for {config_id} --\n", flush=True)

    print(f"\ndone. see {conf_path}, {prop_path}, and {qual_dir}/")


if __name__ == "__main__":
    main()
