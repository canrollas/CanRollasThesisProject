#!/usr/bin/env python3
"""Experiment 7, Step 9 (Chapter 8): evaluation and error analysis.

Pure inference on the already-trained Phase A checkpoints (condition A --
predicted ROI, no retraining here). Per fold:
  - accuracy, macro-F1, MAE, QWK (cross-checks scripts/train.py's own
    exp7_results.csv row for the same fold)
  - per-class precision/recall/F1, confusion matrix
  - number of failed wound localizations in that fold's test split
  - total and trainable parameters
Then, pooled and per-fold:
  - paired comparison vs. the RGB baseline (correct/correct, correct/
    incorrect, incorrect/correct, incorrect/incorrect), matched by
    filename stem (same approach as exp6's complementarity_analysis.py)
  - whether fusion disagreements concentrate on stage 1-vs-2, 2-vs-3, or
    3-vs-4 confusions specifically
  - metrics split by correctly-localized vs. failed-localization images
    (full-dataset metrics above remain the primary number either way)
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.dataset import StageDataset  # noqa: E402
from src.metrics import compute_metrics, predict_labels  # noqa: E402
from src.model import build_model as build_rgb_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from dataset import ROIStageDataset, load_roi_manifest  # noqa: E402
from engine import _boxes_to_list  # noqa: E402
from model import build_model, count_params_m  # noqa: E402

CLASS_ORDER = ["stage_1", "stage_2", "stage_3", "stage_4"]
CONFIG_ID = "mit_b1__corn__global_local_roi_fusion"
RGB_CKPT_DIR = REPO_ROOT / "task_3" / "results" / "task_3_outputs" / "checkpoints"


@torch.no_grad()
def rgb_baseline_predictions(fold_idx, data_root, image_size, device):
    folds = build_folds(data_root, 3, 0.1, 42)
    test_samples = folds[fold_idx]["test"]
    ds = StageDataset(test_samples, data_root, image_size, transform=None)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    model = build_rgb_model("mit_b1", None, "corn", num_classes=4).to(device)
    model.load_state_dict(torch.load(RGB_CKPT_DIR / f"mit_b1__corn_fold{fold_idx}.pt", map_location=device))
    model.eval()

    preds, trues = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        preds.extend(predict_labels(logits, "corn").cpu().tolist())
        trues.extend(targets.tolist())
    stems = [Path(rel).stem for rel, _ in test_samples]
    return {s: (t, p) for s, t, p in zip(stems, trues, preds)}


@torch.no_grad()
def fusion_predictions(fold_idx, base_cfg, data_root, roi_lookup, device):
    folds = build_folds(data_root, 3, 0.1, 42)
    test_samples = folds[fold_idx]["test"]
    ds = ROIStageDataset(test_samples, data_root, roi_lookup, base_cfg["image_size"], augment=None)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_model(
        base_cfg["encoder"], None, num_classes=4,
        roi_output_size=base_cfg["roi_output_size"], local_dim=base_cfg["local_dim"],
    ).to(device)
    ckpt_path = EXP_DIR / "checkpoints" / f"{CONFIG_ID}_fold{fold_idx}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    params_m = count_params_m(model)
    trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    preds, trues, stems_all, detected_all = [], [], [], []
    for rgb, boxes, targets, stems in loader:
        boxes_list = _boxes_to_list(boxes.to(device))
        logits, _ = model(rgb.to(device), boxes_list)
        preds.extend(predict_labels(logits, "corn").cpu().tolist())
        trues.extend(targets.tolist())
        stems_all.extend(stems)
        detected_all.extend(roi_lookup[s]["detected"] for s in stems)
    return (
        {s: (t, p, d) for s, t, p, d in zip(stems_all, trues, preds, detected_all)},
        params_m, trainable_m,
    )


def report_fold(fold_idx, fusion_map, rgb_map):
    y_true = np.array([v[0] for v in fusion_map.values()])
    y_pred = np.array([v[1] for v in fusion_map.values()])
    detected = np.array([v[2] for v in fusion_map.values()])

    m = compute_metrics(y_true.tolist(), y_pred.tolist(), 4)
    n_failed = int((~detected).sum())
    print(f"\n{'='*70}\nFOLD {fold_idx}  (n={len(y_true)}, failed localizations={n_failed})\n{'='*70}")
    print(f"accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} mae={m['mae']:.4f} qwk={m['qwk']:.4f}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0,
    )
    print(f"\n{'class':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for c in range(4):
        print(f"{CLASS_ORDER[c]:<10}{precision[c]:>10.4f}{recall[c]:>10.4f}{f1[c]:>10.4f}{support[c]:>10}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print("\nconfusion matrix (rows=true, cols=pred):")
    print("        " + "".join(f"{c:>10}" for c in CLASS_ORDER))
    for i, row in enumerate(cm):
        print(f"{CLASS_ORDER[i]:<8}" + "".join(f"{v:>10}" for v in row))

    for subset_name, mask in [("correctly localized", detected), ("failed localization", ~detected)]:
        if mask.sum() == 0:
            continue
        sm = compute_metrics(y_true[mask].tolist(), y_pred[mask].tolist(), 4)
        print(f"\n[{subset_name}] n={int(mask.sum())} accuracy={sm['accuracy']:.4f} "
              f"macro_f1={sm['macro_f1']:.4f} qwk={sm['qwk']:.4f}")

    # paired comparison vs RGB baseline
    stems = list(fusion_map.keys())
    assert set(stems) == set(rgb_map.keys())
    both_correct = rgb_only_correct = fusion_only_correct = both_wrong = 0
    stage_pair_flips = Counter()
    for s in stems:
        true, fusion_pred, _ = fusion_map[s]
        true_rgb, rgb_pred = rgb_map[s]
        assert true == true_rgb
        rgb_ok, fusion_ok = rgb_pred == true, fusion_pred == true
        if rgb_ok and fusion_ok:
            both_correct += 1
        elif rgb_ok and not fusion_ok:
            rgb_only_correct += 1
        elif not rgb_ok and fusion_ok:
            fusion_only_correct += 1
        else:
            both_wrong += 1
        if rgb_pred != fusion_pred:
            pair = tuple(sorted((rgb_pred, fusion_pred)))
            if pair in [(0, 1), (1, 2), (2, 3)]:
                stage_pair_flips[pair] += 1

    n = len(stems)
    print(f"\npaired comparison vs RGB baseline (n={n}):")
    print(f"  RGB correct / Fusion correct     = {both_correct} ({100*both_correct/n:.1f}%)")
    print(f"  RGB correct / Fusion incorrect   = {rgb_only_correct} ({100*rgb_only_correct/n:.1f}%)")
    print(f"  RGB incorrect / Fusion correct   = {fusion_only_correct} ({100*fusion_only_correct/n:.1f}%)")
    print(f"  RGB incorrect / Fusion incorrect = {both_wrong} ({100*both_wrong/n:.1f}%)")
    print(f"  prediction disagreements concentrated at stage boundary (1-2,2-3,3-4): {dict(stage_pair_flips)}")

    return {
        "fold": fold_idx, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "mae": m["mae"], "qwk": m["qwk"],
        "n_failed_localizations": n_failed, "both_correct": both_correct, "rgb_only_correct": rgb_only_correct,
        "fusion_only_correct": fusion_only_correct, "both_wrong": both_wrong, "n": n,
    }


def main():
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = EXP_DIR / base_cfg["data_root"]
    roi_lookup = load_roi_manifest(EXP_DIR / base_cfg["roi_manifest_csv"])
    device = get_device()
    print(f"device={device}")

    pooled = Counter()
    params_m = trainable_m = None
    for fold_idx in range(3):
        ckpt_path = EXP_DIR / "checkpoints" / f"{CONFIG_ID}_fold{fold_idx}.pt"
        if not ckpt_path.is_file():
            print(f"fold{fold_idx}: SKIP, no checkpoint")
            continue
        fusion_map, params_m, trainable_m = fusion_predictions(fold_idx, base_cfg, data_root, roi_lookup, device)
        rgb_map = rgb_baseline_predictions(fold_idx, data_root, base_cfg["image_size"], device)
        summary = report_fold(fold_idx, fusion_map, rgb_map)
        for k in ("both_correct", "rgb_only_correct", "fusion_only_correct", "both_wrong", "n"):
            pooled[k] += summary[k]

    if pooled["n"]:
        print(f"\n{'='*70}\nPOOLED ACROSS ALL FOLDS (n={pooled['n']})\n{'='*70}")
        for k in ("both_correct", "rgb_only_correct", "fusion_only_correct", "both_wrong"):
            print(f"  {k} = {pooled[k]} ({100*pooled[k]/pooled['n']:.1f}%)")

    if params_m is not None:
        print(f"\nmodel size: {params_m:.3f}M total, {trainable_m:.3f}M trainable (Phase A)")


if __name__ == "__main__":
    main()
