#!/usr/bin/env python3
"""Builds the exp12 comparison table: ImageNet MiT-B1+CORN (size
reference, NOT retrained), ImageNet MiT-B2+CORN (fair-size control --
ALREADY present in task_3's own ablation grid, NOT retrained here -- see
README), and Task2-init MiT-B2+CORN (this experiment). Reads task_3's own
ablation_results.csv directly; never writes to it.
"""
import csv
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]

TASK3_RESULTS = REPO_ROOT / "task_3/results/task_3_outputs/results/ablation_results.csv"
EXP12_RESULTS = EXP_DIR / "results/exp12_results.csv"

REFERENCE_CONFIGS = ["mit_b1__corn", "mit_b2__corn"]
EXP12_CONFIG = "mit_b2_task2init__corn"
METRICS = ["accuracy", "macro_f1", "mae", "qwk"]


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows, config_id):
    sub = sorted((r for r in rows if r["config_id"] == config_id), key=lambda r: int(r["fold"]))
    out = {"config_id": config_id, "n_folds": len(sub)}
    for m in METRICS:
        vals = np.array([float(r[m]) for r in sub])
        out[f"{m}_mean"] = round(float(vals.mean()), 4) if len(vals) else None
        out[f"{m}_std"] = round(float(vals.std()), 4) if len(vals) else None
    out["per_fold"] = {int(r["fold"]): {m: float(r[m]) for m in METRICS} for r in sub}
    return out


def main():
    task3_rows = read_rows(TASK3_RESULTS)
    exp12_rows = read_rows(EXP12_RESULTS)

    summaries = [summarize(task3_rows, cid) for cid in REFERENCE_CONFIGS]
    summaries.append(summarize(exp12_rows, EXP12_CONFIG))

    print(f"{'config_id':28s} n  " + "  ".join(f"{m:>17s}" for m in METRICS))
    for s in summaries:
        if s["n_folds"] == 0:
            print(f"{s['config_id']:28s} -- not found / not yet run")
            continue
        print(
            f"{s['config_id']:28s} {s['n_folds']}  "
            + "  ".join(f"{s[f'{m}_mean']:.4f}+-{s[f'{m}_std']:.4f}" for m in METRICS)
        )

    mit_b2_imagenet = next(s for s in summaries if s["config_id"] == "mit_b2__corn")
    mit_b2_task2init = next(s for s in summaries if s["config_id"] == EXP12_CONFIG)
    if mit_b2_imagenet["n_folds"] and mit_b2_task2init["n_folds"]:
        print("\nPer-fold delta (Task2-init minus ImageNet MiT-B2), same fold index = same held-out test images:")
        for fold in sorted(mit_b2_imagenet["per_fold"]):
            if fold not in mit_b2_task2init["per_fold"]:
                continue
            a = mit_b2_imagenet["per_fold"][fold]
            b = mit_b2_task2init["per_fold"][fold]
            deltas = "  ".join(f"{m}:{b[m] - a[m]:+.4f}" for m in METRICS)
            print(f"  fold{fold}: {deltas}")


if __name__ == "__main__":
    main()
