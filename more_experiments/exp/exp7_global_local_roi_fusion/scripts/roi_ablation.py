#!/usr/bin/env python3
"""Experiment 7, Step 8 (Chapter 8): ROI contribution ablation.

Inference-only (no retraining) on the ALREADY-TRAINED Phase A checkpoints,
per fold. Three conditions, same trained weights:

  A -- Predicted ROI: the actual Task 1 predicted wound bounding boxes
       (results/roi_manifest.csv, as trained on).
  B -- Full-Image ROI: every box replaced with [0,0,224,224], global RGB
       input unchanged.
  C -- Shuffled ROI: each image's box reassigned from a DIFFERENT image in
       the same fold's test split (a fixed random permutation with no
       fixed points, independent of label), global RGB input unchanged.

Reports accuracy/macro_f1/mae/qwk for each condition, prediction-flip
counts vs. condition A, and -- per the brief's explicit caveat -- the IoU
between each image's shuffled-in box (condition C) and its OWN real
predicted box, since a shuffled box can coincidentally still overlap the
actual wound (two wound photos are often center-framed and similarly
sized) -- this is reported as a limitation on condition C's interpretation,
not swept under the rug.
"""
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.metrics import compute_metrics, predict_labels  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from dataset import ROIStageDataset, load_roi_manifest  # noqa: E402
from engine import _boxes_to_list  # noqa: E402
from model import build_model  # noqa: E402

CONFIG_ID = "mit_b1__corn__global_local_roi_fusion"


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def make_shuffled_override(test_samples, roi_lookup, seed=42):
    """Derangement (no fixed points) over the fold's test stems; each
    image's box comes from a DIFFERENT image, independent of its own label
    (not filtered by stage)."""
    import random
    stems = [Path(rel).stem for rel, _ in test_samples]
    rng = random.Random(seed)
    perm = stems[:]
    for _ in range(1000):
        rng.shuffle(perm)
        if all(a != b for a, b in zip(stems, perm)):
            break
    return {stem: roi_lookup[donor]["box"] for stem, donor in zip(stems, perm)}, dict(zip(stems, perm))


@torch.no_grad()
def run_condition(model, samples, data_root, roi_lookup, image_size, device, box_override=None):
    ds = ROIStageDataset(samples, data_root, roi_lookup, image_size, augment=None, box_override=box_override)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    preds, trues, stems_all = [], [], []
    for rgb, boxes, targets, stems in loader:
        rgb = rgb.to(device)
        boxes = _boxes_to_list(boxes.to(device))
        logits, _ = model(rgb, boxes)
        preds.extend(predict_labels(logits, "corn").cpu().tolist())
        trues.extend(targets.tolist())
        stems_all.extend(stems)
    return stems_all, trues, preds


def main():
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = EXP_DIR / base_cfg["data_root"]
    roi_lookup = load_roi_manifest(EXP_DIR / base_cfg["roi_manifest_csv"])
    device = get_device()
    print(f"device={device}")

    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])

    rows = []
    for fold_idx, fold in enumerate(folds):
        ckpt_path = EXP_DIR / "checkpoints" / f"{CONFIG_ID}_fold{fold_idx}.pt"
        if not ckpt_path.is_file():
            print(f"fold{fold_idx}: SKIP, no checkpoint at {ckpt_path}")
            continue

        model = build_model(
            base_cfg["encoder"], None, num_classes=len(base_cfg["classes"]),
            roi_output_size=base_cfg["roi_output_size"], local_dim=base_cfg["local_dim"],
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        test_samples = fold["test"]
        full_image_override = {Path(rel).stem: [0.0, 0.0, 224.0, 224.0] for rel, _ in test_samples}
        shuffled_override, perm_map = make_shuffled_override(test_samples, roi_lookup, seed=base_cfg["seed"])

        stems_a, trues, preds_a = run_condition(model, test_samples, data_root, roi_lookup, base_cfg["image_size"], device, None)
        _, _, preds_b = run_condition(model, test_samples, data_root, roi_lookup, base_cfg["image_size"], device, full_image_override)
        _, _, preds_c = run_condition(model, test_samples, data_root, roi_lookup, base_cfg["image_size"], device, shuffled_override)

        m_a = compute_metrics(trues, preds_a, len(base_cfg["classes"]))
        m_b = compute_metrics(trues, preds_b, len(base_cfg["classes"]))
        m_c = compute_metrics(trues, preds_c, len(base_cfg["classes"]))

        flips_ab = sum(1 for pa, pb in zip(preds_a, preds_b) if pa != pb)
        flips_ac = sum(1 for pa, pc in zip(preds_a, preds_c) if pa != pc)

        ious = [box_iou(roi_lookup[s]["box"], roi_lookup[perm_map[s]]["box"]) for s in stems_a]
        n_overlap = sum(1 for v in ious if v > 0)
        mean_iou = sum(ious) / len(ious) if ious else 0.0

        print(f"\nfold{fold_idx} (n={len(stems_a)}):")
        print(f"  {'condition':<14}{'accuracy':>10}{'macro_f1':>10}{'mae':>8}{'qwk':>8}")
        for name, m in [("A-predicted", m_a), ("B-full_image", m_b), ("C-shuffled", m_c)]:
            print(f"  {name:<14}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}{m['mae']:>8.4f}{m['qwk']:>8.4f}")
        print(f"  prediction flips vs A: B={flips_ab}/{len(stems_a)}  C={flips_ac}/{len(stems_a)}")
        print(f"  [LIMITATION] shuffled-box vs own-real-box overlap: {n_overlap}/{len(stems_a)} have IoU>0, mean IoU={mean_iou:.4f}")

        for name, m, flips in [("A-predicted", m_a, None), ("B-full_image", m_b, flips_ab), ("C-shuffled", m_c, flips_ac)]:
            rows.append({
                "fold": fold_idx, "condition": name, "n": len(stems_a),
                "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4),
                "mae": round(m["mae"], 4), "qwk": round(m["qwk"], 4),
                "prediction_flips_vs_A": flips if flips is not None else 0,
                "shuffled_box_mean_iou_vs_own_real_box": round(mean_iou, 4) if name == "C-shuffled" else "",
                "shuffled_box_overlap_fraction": round(n_overlap / len(stems_a), 4) if name == "C-shuffled" else "",
            })

    out_path = EXP_DIR / "results" / "roi_ablation_results.csv"
    with open(out_path, "w", newline="") as f:
        fieldnames = ["fold", "condition", "n", "accuracy", "macro_f1", "mae", "qwk",
                      "prediction_flips_vs_A", "shuffled_box_mean_iou_vs_own_real_box", "shuffled_box_overlap_fraction"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
