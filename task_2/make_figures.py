#!/usr/bin/env python3
"""Builds the mIoU/mDice + per-class-IoU comparison bar chart from
results/ablation_summary.csv, in the same two-panel style as the
segmentation-benchmarking project's scripts/evaluate.py plots.

Unlike that script, this doesn't re-run inference or bootstrap an
uncertainty estimate from a single train/test split: run_ablation.py already
evaluates every (config, fold) on its own held-out test fold, and
write_summary() already aggregates mean +/- std across folds into
ablation_summary.csv -- real cross-fold variance, which is the more
principled uncertainty estimate for a k-fold protocol than a bootstrap over
one fold's predictions.

Usage:
    python make_figures.py                        # every config in ablation_summary.csv
    python make_figures.py --top 10                # only the 10 best by mIoU
    python make_figures.py --configs manet__mit_b2__imagenet,unet__mit_b1__imagenet
    python make_figures.py --output results/figures/top10.png
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR))

from src.utils import load_yaml  # noqa: E402

CLASS_COLORS = ["#e74c3c", "#f1c40f", "#8e44ad", "#1abc9c", "#2ecc71"]
MIOU_COLOR = "#2ecc71"
MDICE_COLOR = "#3498db"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", default=None, help="defaults to <checkpoint_dir's sibling> ablation_summary.csv")
    p.add_argument("--configs", default=None, help="comma-separated config_id allowlist")
    p.add_argument("--top", type=int, default=None, help="keep only the N best configs by mIoU")
    p.add_argument("--output", default=None, help="defaults to results/ablation_figure.png")
    return p.parse_args()


def load_rows(summary_csv, configs_arg, top):
    with open(summary_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{summary_csv} is empty")

    if configs_arg:
        wanted = [c.strip() for c in configs_arg.split(",") if c.strip()]
        by_id = {r["config_id"]: r for r in rows}
        missing = [c for c in wanted if c not in by_id]
        if missing:
            raise SystemExit(f"--configs: not in {summary_csv.name}: {missing}")
        rows = [by_id[c] for c in wanted]
    else:
        rows.sort(key=lambda r: float(r["miou_mean"]), reverse=True)

    if top:
        rows = rows[:top]
    if not rows:
        raise SystemExit("nothing to plot")
    return rows


def main():
    args = parse_args()
    base_cfg = load_yaml(TASK_DIR / "configs" / "base.yaml")
    class_names = base_cfg["classes"]

    summary_csv = Path(args.summary_csv) if args.summary_csv else TASK_DIR / "results" / "ablation_summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"{summary_csv} not found — run run_ablation.py first")

    rows = load_rows(summary_csv, args.configs, args.top)

    labels = [f"{r['architecture']}\n{r['encoder']}\n({r['weights']})" for r in rows]
    n_models = len(rows)
    n_classes = len(class_names)

    miou = np.array([float(r["miou_mean"]) for r in rows])
    miou_err = np.array([float(r["miou_std"]) for r in rows])
    mdice = np.array([float(r["mdice_mean"]) for r in rows])
    mdice_err = np.array([float(r["mdice_std"]) for r in rows])
    per_cls_iou = [[float(r[f"iou_{cn}_mean"]) for r in rows] for cn in class_names]
    per_cls_iou_err = [[float(r[f"iou_{cn}_std"]) for r in rows] for cn in class_names]

    n_folds = rows[0].get("n_folds", "?")
    x = np.arange(n_models)
    bar_w = 0.13

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_models * 0.9), 12), gridspec_kw={"hspace": 0.55})
    fig.suptitle(
        f"Test Set Evaluation — wound-tissue-segmentation  "
        f"({n_folds}-fold CV, mean ± std across folds, {n_models} config{'s' if n_models != 1 else ''})",
        fontsize=14, fontweight="bold",
    )

    err_kw = dict(capsize=3, capthick=1.0, elinewidth=1.0, ecolor="black", alpha=0.7)

    ax = axes[0]
    ax.bar(x - bar_w / 2, miou, bar_w, label="mIoU", color=MIOU_COLOR, alpha=0.85, yerr=miou_err, error_kw=err_kw)
    ax.bar(x + bar_w / 2, mdice, bar_w, label="mDice", color=MDICE_COLOR, alpha=0.85, yerr=mdice_err, error_kw=err_kw)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, ha="center")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.10)
    ax.set_title("Mean IoU & Mean Dice (sorted by mIoU ↓)")
    ax.legend(loc="upper right")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    for xi, (mi, mi_s, md, md_s) in enumerate(zip(miou, miou_err, mdice, mdice_err)):
        ax.text(xi - bar_w / 2, mi + mi_s + 0.015, f"{mi:.3f}\n±{mi_s:.3f}", ha="center", va="bottom", fontsize=4.5)
        ax.text(xi + bar_w / 2, md + md_s + 0.015, f"{md:.3f}\n±{md_s:.3f}", ha="center", va="bottom", fontsize=4.5)

    ax2 = axes[1]
    offsets = np.linspace(-(n_classes - 1) * bar_w / 2, (n_classes - 1) * bar_w / 2, n_classes)
    for c, (cls_scores, cls_err, color, offset) in enumerate(zip(per_cls_iou, per_cls_iou_err, CLASS_COLORS, offsets)):
        cls_scores = np.array(cls_scores)
        cls_err = np.array(cls_err)
        ax2.bar(x + offset, cls_scores, bar_w, color=color, alpha=0.80, label=class_names[c], yerr=cls_err, error_kw=err_kw)
        for xi, (v, s) in enumerate(zip(cls_scores, cls_err)):
            ax2.text(xi + offset, v + s + 0.012, f"{v:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=4.0)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6.5, ha="center")
    ax2.set_ylabel("IoU")
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Per-Class IoU")
    ax2.legend(loc="upper right")
    ax2.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)

    out_path = Path(args.output) if args.output else TASK_DIR / "results" / "ablation_figure.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out_path}")


if __name__ == "__main__":
    main()
