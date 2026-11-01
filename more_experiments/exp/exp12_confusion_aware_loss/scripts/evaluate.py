#!/usr/bin/env python3
"""Aggregates one variant's test-set predictions (results/predictions/
{config_id}_fold{0,1,2}.csv, written by train.py --mode final) into the
full comparison the brief's Section 6 requires: per-fold and pooled
confusion matrices, per-stage precision/recall/F1, every directed
stage-pair error count, Stage 3->4 / 4->3 counts, adjacent vs non-adjacent
error totals, and a paired fixed/regressed comparison against a baseline
predictions file, matched by image path.

Usage:
  python evaluate.py --config-id mit_b1__corn__exp12B_lam0.1_alpha1
  python evaluate.py --config-id mit_b1__corn__exp12_baseline \
      --baseline-predictions none   # the baseline has nothing to compare against itself

Default baseline: results/predictions/mit_b1__corn__exp12_baseline_fold{i}.csv
(this script's OWN reproduction of the Chapter 7 baseline, trained by this
same train.py/engine.py/losses.py codebase under variant='baseline' -- an
exact apples-to-apples comparison, not merely a historically similar run).
Pass --baseline-predictions to point at a different file (e.g. the
original more_experiments/reports/figures/mit_b1_corn_test_explanations/
all_predictions.csv) or 'none' to skip the paired comparison.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]

CLASS_LABELS = ["stage_1", "stage_2", "stage_3", "stage_4"]
NUM_CLASSES = 4


def load_predictions(config_id):
    rows = []
    pred_dir = EXP_DIR / "results" / "predictions"
    for fold in range(3):
        path = pred_dir / f"{config_id}_fold{fold}.csv"
        if not path.is_file():
            print(f"WARNING: missing {path}, skipping fold {fold}")
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                r["true_label"] = int(r["true_label"])
                r["pred_label"] = int(r["pred_label"])
                r["fold"] = int(r["fold"])
                rows.append(r)
    if not rows:
        raise SystemExit(f"no prediction files found for config_id={config_id} under {pred_dir}")
    return rows


def confusion_counts(rows):
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["pred_label"] for r in rows]
    return confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))


def metrics_block(rows):
    y_true = np.array([r["true_label"] for r in rows])
    y_pred = np.array([r["pred_label"] for r in rows])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0,
    )
    return {
        "n": len(rows),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(NUM_CLASSES)))),
        "per_stage": {
            CLASS_LABELS[i]: {
                "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            } for i in range(NUM_CLASSES)
        },
    }


def stage_pair_errors(rows):
    errors = [(r["true_label"], r["pred_label"]) for r in rows if r["true_label"] != r["pred_label"]]
    counts = Counter(errors)
    adjacent = sum(c for (t, p), c in counts.items() if abs(t - p) == 1)
    non_adjacent = sum(c for (t, p), c in counts.items() if abs(t - p) >= 2)
    return {
        "total_errors": len(errors),
        "adjacent_errors": adjacent,
        "non_adjacent_errors": non_adjacent,
        "stage3_to_stage4": counts.get((2, 3), 0),
        "stage4_to_stage3": counts.get((3, 2), 0),
        "all_directed_pairs": {
            f"{CLASS_LABELS[t]}->{CLASS_LABELS[p]}": c for (t, p), c in sorted(counts.items(), key=lambda kv: -kv[1])
        },
    }


def paired_comparison(rows, baseline_rows):
    baseline_by_path = {r["path"]: r for r in baseline_rows}
    fixed, regressed, both_correct, both_wrong, missing = 0, 0, 0, 0, 0
    fixed_rows, regressed_rows = [], []
    for r in rows:
        base = baseline_by_path.get(r["path"])
        if base is None:
            missing += 1
            continue
        base_correct = int(base["true_label"]) == int(base["pred_label"])
        new_correct = r["true_label"] == r["pred_label"]
        if base_correct and not new_correct:
            regressed += 1
            regressed_rows.append(r["path"])
        elif not base_correct and new_correct:
            fixed += 1
            fixed_rows.append(r["path"])
        elif base_correct and new_correct:
            both_correct += 1
        else:
            both_wrong += 1
    return {
        "n_matched": len(rows) - missing, "n_missing_from_baseline": missing,
        "fixed": fixed, "regressed": regressed,
        "both_correct": both_correct, "both_wrong": both_wrong,
        "net_change": fixed - regressed,
        "fixed_paths": fixed_rows, "regressed_paths": regressed_rows,
    }


def save_confusion_figure(counts, out_path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(counts, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([c.replace("stage_", "S") for c in CLASS_LABELS])
    ax.set_yticklabels([c.replace("stage_", "S") for c in CLASS_LABELS])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                     color="white" if counts[i, j] > counts.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_stage_pair_figure(pair_counts, out_path, title):
    items = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:10]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(len(labels)), values[::-1])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("error count")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-id", required=True)
    p.add_argument("--baseline-predictions", default="default",
                    help="'default' = this codebase's own exp12_baseline predictions, "
                         "'none' = skip paired comparison, or an explicit CSV path")
    args = p.parse_args()

    rows = load_predictions(args.config_id)
    out_dir = EXP_DIR / "results" / "reports" / args.config_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"config_id": args.config_id, "pooled": metrics_block(rows)}
    report["pooled"]["stage_pairs"] = stage_pair_errors(rows)

    pooled_cm = confusion_counts(rows)
    report["pooled"]["confusion_matrix"] = pooled_cm.tolist()
    save_confusion_figure(pooled_cm, out_dir / "confusion_matrix_pooled.png",
                           f"{args.config_id} -- pooled (n={len(rows)})")
    save_stage_pair_figure(report["pooled"]["stage_pairs"]["all_directed_pairs"],
                            out_dir / "stage_pair_errors.png", f"{args.config_id} -- directed stage-pair errors")

    report["per_fold"] = {}
    rows_by_fold = defaultdict(list)
    for r in rows:
        rows_by_fold[r["fold"]].append(r)
    for fold, fold_rows in sorted(rows_by_fold.items()):
        fold_report = metrics_block(fold_rows)
        fold_report["stage_pairs"] = stage_pair_errors(fold_rows)
        fold_cm = confusion_counts(fold_rows)
        fold_report["confusion_matrix"] = fold_cm.tolist()
        save_confusion_figure(fold_cm, out_dir / f"confusion_matrix_fold{fold}.png",
                               f"{args.config_id} -- fold {fold} (n={len(fold_rows)})")
        report["per_fold"][fold] = fold_report

    if args.baseline_predictions != "none":
        if args.baseline_predictions == "default":
            baseline_rows = load_predictions("mit_b1__corn__exp12_baseline")
        else:
            with open(args.baseline_predictions, newline="") as f:
                baseline_rows = list(csv.DictReader(f))
                for r in baseline_rows:
                    r["true_label"] = int(r.get("true_label", r.get("true_stage", "").replace("stage_", "")) or 0)
                    r["pred_label"] = int(r.get("pred_label", r.get("pred_stage", "").replace("stage_", "")) or 0)
        report["paired_vs_baseline"] = paired_comparison(rows, baseline_rows)

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "accuracy", "macro_f1", "mae", "qwk", "n"])
        writer.writerow(["pooled", report["pooled"]["accuracy"], report["pooled"]["macro_f1"],
                          report["pooled"]["mae"], report["pooled"]["qwk"], report["pooled"]["n"]])
        for fold, fr in sorted(report["per_fold"].items()):
            writer.writerow([f"fold{fold}", fr["accuracy"], fr["macro_f1"], fr["mae"], fr["qwk"], fr["n"]])

    print(f"wrote {out_dir}")
    print(json.dumps({k: v for k, v in report["pooled"].items() if k not in ("confusion_matrix", "per_stage")}, indent=2))


if __name__ == "__main__":
    main()
