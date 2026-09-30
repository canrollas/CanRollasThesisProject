import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


RESULT_FIELDS = [
    "config_id", "architecture", "encoder", "weights", "fold",
    "params_m", "miou", "mdice", "iou_bg", "iou_skin", "iou_wound",
    "dice_bg", "dice_skin", "dice_wound", "train_time_s",
]


def load_completed_keys(results_csv):
    completed = set()
    if not os.path.exists(results_csv):
        return completed
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row["config_id"], row["fold"]))
    return completed


def append_result(results_csv, row):
    path = Path(results_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


FAILURE_FIELDS = ["config_id", "architecture", "encoder", "weights", "fold", "error", "timestamp"]


def append_failure(failures_csv, row):
    """Logs a (config, fold) that raised instead of crashing the whole grid,
    so it's visible and can be investigated/re-run individually afterwards."""
    path = Path(failures_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


SUMMARY_NUMERIC_FIELDS = [
    "miou", "mdice", "iou_bg", "iou_skin", "iou_wound",
    "dice_bg", "dice_skin", "dice_wound", "train_time_s",
]


def write_summary(results_csv, summary_csv):
    """Rewrites a per-config mean+-std rollup over completed folds, matching
    the paper's Table 4 reporting format. Safe to call after every fold —
    it re-derives everything from results_csv, so an interrupted run leaves
    a summary consistent with whatever folds actually finished.
    """
    results_path = Path(results_csv)
    if not results_path.exists():
        return

    with open(results_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    groups = {}
    for row in rows:
        groups.setdefault(row["config_id"], []).append(row)

    summary_rows = []
    for config_id, group_rows in groups.items():
        first = group_rows[0]
        summary = {
            "config_id": config_id,
            "architecture": first["architecture"],
            "encoder": first["encoder"],
            "weights": first["weights"],
            "params_m": first["params_m"],
            "n_folds": len(group_rows),
        }
        for field in SUMMARY_NUMERIC_FIELDS:
            values = np.array([float(r[field]) for r in group_rows])
            summary[f"{field}_mean"] = round(float(values.mean()), 4)
            summary[f"{field}_std"] = round(float(values.std()), 4)
        summary_rows.append(summary)

    summary_rows.sort(key=lambda r: r["miou_mean"], reverse=True)

    summary_path = Path(summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
