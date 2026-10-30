#!/usr/bin/env python3
"""Builds the exp13 comparison table: ImageNet MiT-B1+CORN and
ImageNet MiT-B2+CORN (both read directly from task_3's own
ablation_results.csv, NOT retrained here) vs. DINOv2 ViT-L/14+CORN,
frozen and finetune modes (this experiment).
"""
import csv
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]

TASK3_RESULTS = REPO_ROOT / "task_3/results/task_3_outputs/results/ablation_results.csv"
EXP13_RESULTS = EXP_DIR / "results/exp13_results.csv"

REFERENCE_CONFIGS = ["mit_b1__corn", "mit_b2__corn"]
EXP13_CONFIGS = ["dinov2_vitl14__corn_frozen", "dinov2_vitl14__corn_finetune"]
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
    return out


def main():
    task3_rows = read_rows(TASK3_RESULTS)
    exp13_rows = read_rows(EXP13_RESULTS)

    summaries = [summarize(task3_rows, cid) for cid in REFERENCE_CONFIGS]
    summaries += [summarize(exp13_rows, cid) for cid in EXP13_CONFIGS]

    print(f"{'config_id':32s} n  " + "  ".join(f"{m:>17s}" for m in METRICS))
    for s in summaries:
        if s["n_folds"] == 0:
            print(f"{s['config_id']:32s} -- not found / not yet run")
            continue
        print(
            f"{s['config_id']:32s} {s['n_folds']}  "
            + "  ".join(f"{s[f'{m}_mean']:.4f}+-{s[f'{m}_std']:.4f}" for m in METRICS)
        )


if __name__ == "__main__":
    main()
