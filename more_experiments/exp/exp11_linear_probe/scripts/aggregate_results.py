#!/usr/bin/env python3
"""Experiment 11 (Chapter 8): linear-probe diagnostic for CORN's errors
-- Step 6, three-fold aggregation.

Reads what extract_features.py / train_probe.py / compare_errors.py
already wrote per fold (no recomputation of anything, no re-touching of
test data -- this script only aggregates numbers that already exist on
disk) and reports, across all 3 folds:

  1. Accuracy / macro-F1 / MAE / QWK, mean +- population std (same
     convention as task_3/src/utils.py's write_summary -- ddof=0), for
     CORN itself, the `final`-level probe, and the `stage3`-level probe.
  2. Per-fold and summed-across-folds error-breakdown counts (both
     correct / probe rescues / probe breaks / both wrong) for each probe
     level.
  3. The true=stage_3/CORN-predicted=stage_4 cell specifically, per fold
     and summed, since the proposal calls it out by name.

Writes results/aggregate_summary.json (everything) and
results/aggregate_summary.csv (the metrics table only, for a quick diff
against task_3's own ablation_results.csv-style format).
"""
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.metrics import compute_metrics  # noqa: E402

FOLDS = (0, 1, 2)
METRICS = ("accuracy", "macro_f1", "mae", "qwk")


def corn_test_metrics(fold_dir):
    labels = np.load(fold_dir / "test_labels.npy")
    preds = np.load(fold_dir / "test_corn_preds.npy")
    stats = compute_metrics(labels, preds, num_classes=4)
    stats.pop("confusion")
    return stats


def probe_test_metrics(fold_dir, level):
    with open(fold_dir / f"probe_{level}_summary.json") as f:
        summary = json.load(f)
    return summary["test"]


def mean_std(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())  # population std (ddof=0), matches src/utils.py write_summary


def main():
    results_dir = EXP_DIR / "results"
    per_fold = {"CORN": {}, "probe_final": {}, "probe_stage3": {}}
    error_breakdown = {"final": {}, "stage3": {}}
    stage3_to_4_cell = {"final": {}, "stage3": {}}

    for fold in FOLDS:
        fold_dir = results_dir / f"fold_{fold}"
        per_fold["CORN"][fold] = corn_test_metrics(fold_dir)
        per_fold["probe_final"][fold] = probe_test_metrics(fold_dir, "final")
        per_fold["probe_stage3"][fold] = probe_test_metrics(fold_dir, "stage3")

        for level in ("final", "stage3"):
            with open(fold_dir / f"compare_{level}_summary.json") as f:
                compare = json.load(f)
            error_breakdown[level][fold] = {
                "n_test": compare["n_test"],
                "both_correct": compare["both_correct"],
                "probe_rescues": compare["probe_rescues"],
                "probe_breaks": compare["probe_breaks"],
                "both_wrong": compare["both_wrong"],
                "n_corn_errors": compare["n_corn_errors"],
                "rescue_rate": compare["rescue_rate"],
            }
            named = next(
                (row for row in compare["pair_breakdown"]
                 if row["true"] == "stage_3" and row["corn_pred"] == "stage_4"),
                {"n_corn_errors": 0, "n_probe_rescued": 0, "rescue_rate": 0.0},
            )
            stage3_to_4_cell[level][fold] = named

    metrics_table = {}
    for model_key in ("CORN", "probe_final", "probe_stage3"):
        metrics_table[model_key] = {}
        for metric in METRICS:
            values = [per_fold[model_key][fold][metric] for fold in FOLDS]
            mean, std = mean_std(values)
            metrics_table[model_key][metric] = {
                "mean": round(mean, 4), "std": round(std, 4),
                "per_fold": [round(v, 4) for v in values],
            }

    error_breakdown_summed = {}
    for level in ("final", "stage3"):
        total_rescues = sum(error_breakdown[level][f]["probe_rescues"] for f in FOLDS)
        total_breaks = sum(error_breakdown[level][f]["probe_breaks"] for f in FOLDS)
        total_both_wrong = sum(error_breakdown[level][f]["both_wrong"] for f in FOLDS)
        total_corn_errors = sum(error_breakdown[level][f]["n_corn_errors"] for f in FOLDS)
        error_breakdown_summed[level] = {
            "total_probe_rescues": total_rescues, "total_probe_breaks": total_breaks,
            "total_both_wrong": total_both_wrong, "total_corn_errors": total_corn_errors,
            "pooled_rescue_rate": round(total_rescues / total_corn_errors, 4) if total_corn_errors else None,
        }

    stage3_to_4_summed = {}
    for level in ("final", "stage3"):
        total_n = sum(stage3_to_4_cell[level][f]["n_corn_errors"] for f in FOLDS)
        total_rescued = sum(stage3_to_4_cell[level][f]["n_probe_rescued"] for f in FOLDS)
        stage3_to_4_summed[level] = {
            "total_n": total_n, "total_rescued": total_rescued,
            "pooled_rescue_rate": round(total_rescued / total_n, 4) if total_n else None,
        }

    summary = {
        "folds": list(FOLDS),
        "metrics_mean_std_over_3_folds": metrics_table,
        "error_breakdown_per_fold": error_breakdown,
        "error_breakdown_summed_over_3_folds": error_breakdown_summed,
        "stage3_true_stage4_corn_pred_cell_per_fold": stage3_to_4_cell,
        "stage3_true_stage4_corn_pred_cell_summed": stage3_to_4_summed,
    }
    with open(results_dir / "aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(results_dir / "aggregate_summary.csv", "w") as f:
        f.write("model,metric,mean,std,fold0,fold1,fold2\n")
        for model_key in ("CORN", "probe_final", "probe_stage3"):
            for metric in METRICS:
                row = metrics_table[model_key][metric]
                f.write(f"{model_key},{metric},{row['mean']},{row['std']},"
                        f"{row['per_fold'][0]},{row['per_fold'][1]},{row['per_fold'][2]}\n")

    print("=== accuracy / macro-F1 / MAE / QWK, mean +- std over 3 folds ===")
    for model_key in ("CORN", "probe_final", "probe_stage3"):
        row = metrics_table[model_key]
        print(f"{model_key:>14}: acc={row['accuracy']['mean']:.4f}+-{row['accuracy']['std']:.4f} "
              f"f1={row['macro_f1']['mean']:.4f}+-{row['macro_f1']['std']:.4f} "
              f"mae={row['mae']['mean']:.4f}+-{row['mae']['std']:.4f} "
              f"qwk={row['qwk']['mean']:.4f}+-{row['qwk']['std']:.4f}")

    print("\n=== error breakdown, summed over 3 folds ===")
    for level in ("final", "stage3"):
        s = error_breakdown_summed[level]
        print(f"[{level}] rescued={s['total_probe_rescues']} new_errors={s['total_probe_breaks']} "
              f"common={s['total_both_wrong']} / corn_errors={s['total_corn_errors']} "
              f"pooled_rescue_rate={s['pooled_rescue_rate']}")

    print("\n=== true=stage_3, CORN-predicted=stage_4 cell, summed over 3 folds ===")
    for level in ("final", "stage3"):
        s = stage3_to_4_summed[level]
        print(f"[{level}] n={s['total_n']} rescued={s['total_rescued']} rate={s['pooled_rescue_rate']}")

    print(f"\nwrote {results_dir / 'aggregate_summary.json'}")
    print(f"wrote {results_dir / 'aggregate_summary.csv'}")


if __name__ == "__main__":
    main()
