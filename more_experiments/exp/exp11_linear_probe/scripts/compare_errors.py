#!/usr/bin/env python3
"""Experiment 11 (Chapter 8): linear-probe diagnostic for CORN's errors
-- Step 4/5, the actual comparison this experiment exists to produce.

Joins, on the TEST split only, three things extract_features.py and
train_probe.py already saved for one fold: the true label, CORN's own
prediction (from the real checkpoint), and the linear probe's prediction
(`--level final` by default -- the same 512-dim vector CornHead reads).
Reports the four-way breakdown the proposal asks for (both correct /
probe rescues / probe breaks / both wrong), plus a per-(true,
corn_pred)-pair breakdown of the probe's correction rate among CORN's
own errors, with the true_stage_3-predicted_stage_4 group called out
explicitly (not because it's the only one that matters, but because the
proposal names it as a concrete motivating case -- every other
mislabeled pair is reported in the same table, not suppressed).

This script draws NO conclusion about whether the encoder is "sufficient"
or not from a single fold -- see README.md's section 5 table for why a
single probe run cannot settle that either way.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.dataset import CLASS_ORDER  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--level", choices=("final", "stage3"), default="final")
    args = p.parse_args()

    fold_dir = EXP_DIR / "results" / f"fold_{args.fold}"
    labels = np.load(fold_dir / "test_labels.npy")
    corn_preds = np.load(fold_dir / "test_corn_preds.npy")
    ids = np.load(fold_dir / "test_ids.npy")
    probe_pred_path = fold_dir / f"probe_{args.level}_test_preds.npy"
    if not probe_pred_path.is_file():
        raise SystemExit(f"BLOCKED: {probe_pred_path} not found -- run train_probe.py "
                          f"--fold {args.fold} --level {args.level} first.")
    probe_preds = np.load(probe_pred_path)

    corn_correct = corn_preds == labels
    probe_correct = probe_preds == labels

    both_correct = int((corn_correct & probe_correct).sum())
    probe_rescues = int((~corn_correct & probe_correct).sum())
    probe_breaks = int((corn_correct & ~probe_correct).sum())
    both_wrong = int((~corn_correct & ~probe_correct).sum())
    n_corn_errors = int((~corn_correct).sum())

    rescue_rate = probe_rescues / n_corn_errors if n_corn_errors else float("nan")
    common_error_rate = both_wrong / n_corn_errors if n_corn_errors else float("nan")

    print(f"[fold{args.fold} {args.level}] n_test={len(labels)} "
          f"corn_acc={corn_correct.mean():.4f} probe_acc={probe_correct.mean():.4f}")
    print(f"  both correct:        {both_correct}")
    print(f"  probe rescues:       {probe_rescues}  (CORN wrong, probe right)")
    print(f"  probe breaks:        {probe_breaks}  (CORN right, probe wrong)")
    print(f"  both wrong:          {both_wrong}")
    print(f"  CORN error count:    {n_corn_errors}")
    print(f"  probe rescue rate:   {rescue_rate:.4f}  (of CORN's own errors)")
    print(f"  common error rate:   {common_error_rate:.4f}  (of CORN's own errors)")

    # Per-(true, corn_pred) breakdown among CORN's errors -- true_stage_3
    # predicted_stage_4 is the proposal's named case, called out below,
    # but every pair with at least one error is reported, not just it.
    pair_rows = []
    for true_idx in range(len(CLASS_ORDER)):
        for pred_idx in range(len(CLASS_ORDER)):
            if true_idx == pred_idx:
                continue
            mask = (~corn_correct) & (labels == true_idx) & (corn_preds == pred_idx)
            n = int(mask.sum())
            if n == 0:
                continue
            n_rescued = int((mask & probe_correct).sum())
            pair_rows.append({
                "true": CLASS_ORDER[true_idx], "corn_pred": CLASS_ORDER[pred_idx],
                "n_corn_errors": n, "n_probe_rescued": n_rescued,
                "rescue_rate": round(n_rescued / n, 4),
            })
    pair_rows.sort(key=lambda r: r["n_corn_errors"], reverse=True)

    print("\n  per-(true, corn_pred) breakdown, among CORN's own errors:")
    for row in pair_rows:
        flag = "  <-- named case" if row["true"] == "stage_3" and row["corn_pred"] == "stage_4" else ""
        print(f"    true={row['true']:<8} corn_pred={row['corn_pred']:<8} "
              f"n={row['n_corn_errors']:>3} probe_rescued={row['n_probe_rescued']:>3} "
              f"rescue_rate={row['rescue_rate']:.4f}{flag}")

    # Per-image breakdown for manual inspection (the proposal's "tek tek
    # karşılaştırırız" step) -- every image where CORN and the probe
    # disagree with each other or with the truth.
    per_image_path = fold_dir / f"compare_{args.level}_per_image.csv"
    with open(per_image_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "true", "corn_pred", "probe_pred", "group"])
        for i in range(len(labels)):
            if corn_correct[i] and probe_correct[i]:
                continue  # both right -- not an error-analysis row
            group = (
                "probe_rescue" if (not corn_correct[i] and probe_correct[i]) else
                "probe_break" if (corn_correct[i] and not probe_correct[i]) else
                "both_wrong"
            )
            writer.writerow([ids[i], CLASS_ORDER[labels[i]], CLASS_ORDER[corn_preds[i]],
                              CLASS_ORDER[probe_preds[i]], group])

    summary = {
        "fold": args.fold, "level": args.level, "n_test": int(len(labels)),
        "corn_acc": float(corn_correct.mean()), "probe_acc": float(probe_correct.mean()),
        "both_correct": both_correct, "probe_rescues": probe_rescues,
        "probe_breaks": probe_breaks, "both_wrong": both_wrong,
        "n_corn_errors": n_corn_errors, "rescue_rate": rescue_rate, "common_error_rate": common_error_rate,
        "pair_breakdown": pair_rows,
    }
    summary_path = fold_dir / f"compare_{args.level}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nwrote {per_image_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
