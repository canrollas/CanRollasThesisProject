#!/usr/bin/env python3
"""Builds the frozen, per-fold Exp-12B cost matrices from the EXISTING
baseline mit_b1__corn checkpoints (task_3/results/task_3_outputs/checkpoints/
mit_b1__corn_fold{0,1,2}.pt), evaluated on each fold's own TRAIN-SIDE
VALIDATION split (build_folds(...)[fold]["val"]) -- never the outer test
split, per the brief's "Never compute costs from the outer test fold."

No training happens here: this only runs the already-trained baseline
checkpoint forward on ~109 validation images per fold (val_fraction=0.1 of
the ~980 train+val images), builds a [4,4] true-vs-predicted confusion
count matrix per fold, then calls losses.build_cost_matrix() for each
requested alpha. Fast, CPU/MPS-only, no GPU required.

Output per fold: results/cost_matrices/fold{i}/
  confusion_counts.json     -- raw [4,4] true(rows)xpred(cols) counts
  cost_matrix_alpha{a}.json -- frozen [4,4] cost matrix for that alpha
  confusion_matrix.png      -- figure

These are computed and frozen ONCE, before any Exp-12B training run uses
them -- train.py loads the saved cost_matrix_alpha{a}.json for the
requested fold/alpha rather than recomputing it, so every Exp-12B run in a
given fold uses an identical, already-audited cost matrix.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent  # exp/exp12_confusion_aware_loss/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))
sys.path.insert(0, str(THIS_DIR))

from src.dataset import StageDataset  # noqa: E402
from src.metrics import predict_labels  # noqa: E402
from src.model import build_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402

from losses import build_cost_matrix  # noqa: E402

CLASS_LABELS = ["stage_1", "stage_2", "stage_3", "stage_4"]
ALPHAS = [1.0, 2.0]
BASELINE_CHECKPOINT_DIR = REPO_ROOT / "task_3/results/task_3_outputs/checkpoints"


def load_baseline(fold_idx, device):
    ckpt_path = BASELINE_CHECKPOINT_DIR / f"mit_b1__corn_fold{fold_idx}.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"baseline checkpoint not found: {ckpt_path}")
    # weights=None: the state_dict load below overwrites every weight anyway,
    # so there is no need to also download ImageNet weights first (same
    # convention as exp10_boundary_specific_representation/scripts/model.py's
    # load_rgb_baseline()).
    model = build_model("mit_b1", None, "corn", num_classes=len(CLASS_LABELS))
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def predict_fold_val(model, val_samples, data_root, image_size, batch_size, device):
    ds = StageDataset(val_samples, data_root, image_size, transform=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds = predict_labels(logits, "corn")
            all_true.extend(targets.tolist())
            all_pred.extend(preds.cpu().tolist())
    return np.array(all_true), np.array(all_pred)


def confusion_counts(y_true, y_pred, num_classes=4):
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        counts[t, p] += 1
    return counts


def save_confusion_figure(counts, out_path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(counts, cmap="Blues")
    ax.set_xticks(range(len(CLASS_LABELS)))
    ax.set_yticks(range(len(CLASS_LABELS)))
    ax.set_xticklabels([c.replace("stage_", "S") for c in CLASS_LABELS])
    ax.set_yticklabels([c.replace("stage_", "S") for c in CLASS_LABELS])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                     color="white" if counts[i, j] > counts.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = REPO_ROOT / "datasets" / "wound-stage-classification"
    device = get_device()

    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    out_root = EXP_DIR / "results" / "cost_matrices"
    out_root.mkdir(parents=True, exist_ok=True)

    for fold_idx, fold in enumerate(folds):
        print(f"[fold {fold_idx}] loading baseline checkpoint, evaluating on {len(fold['val'])} validation images")
        model = load_baseline(fold_idx, device)
        y_true, y_pred = predict_fold_val(model, fold["val"], data_root, base_cfg["image_size"],
                                           base_cfg["batch_size"], device)
        counts = confusion_counts(y_true, y_pred)

        fold_dir = out_root / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        with open(fold_dir / "confusion_counts.json", "w") as f:
            json.dump({
                "fold": fold_idx,
                "n_val": len(fold["val"]),
                "source_checkpoint": str((BASELINE_CHECKPOINT_DIR / f"mit_b1__corn_fold{fold_idx}.pt")
                                          .relative_to(REPO_ROOT)),
                "class_order": CLASS_LABELS,
                "confusion_counts": counts.tolist(),
                "accuracy_on_val": float((y_true == y_pred).mean()),
            }, f, indent=2)

        save_confusion_figure(counts, fold_dir / "confusion_matrix.png",
                               f"fold {fold_idx} validation confusion (n={len(fold['val'])})")

        for alpha in ALPHAS:
            cost = build_cost_matrix(counts, alpha=alpha, num_classes=len(CLASS_LABELS))
            with open(fold_dir / f"cost_matrix_alpha{alpha:g}.json", "w") as f:
                json.dump({
                    "fold": fold_idx,
                    "alpha": alpha,
                    "class_order": CLASS_LABELS,
                    "cost_matrix": cost.tolist(),
                }, f, indent=2)

        print(f"[fold {fold_idx}] val_accuracy={float((y_true == y_pred).mean()):.4f} "
              f"confusion_counts=\n{counts}")
        del model

    print(f"wrote cost matrices under {out_root}")


if __name__ == "__main__":
    main()
