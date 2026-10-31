#!/usr/bin/env python3
"""Work Package 4 (advisor extended experiments): Stage 3/4 error audit, step 1.

Builds the case-level table for every TEST-fold image whose true stage is
stage_3 or stage_4 and whose RGB-baseline prediction is also stage_3 or
stage_4 (i.e. a genuine 3-vs-4 case, correct or not) -- true=2/pred=0or1 and
true=3/pred=0or1 rows (18 total, cross-stage mistakes unrelated to the 3-vs-4
question this work package is about) are excluded.

Reuses, rather than re-runs, the already-computed RGB-baseline predictions in
more_experiments/reports/figures/mit_b1_corn_test_explanations/all_predictions.csv
(every test image, all 3 folds, produced by task_3/explain_test_folds.py's own
CORN decode -- see corn_outputs() in task_3/corn_explain.py). No model is
loaded here.

For each case, loads the matching 8-channel tissue array that exp-3 already
generated (datasets/wound-staging-masks-exp-3/<path>.npy, channel order R,G,B,
granulation,slough,necrosis,tendon,bone -- see that dataset's provenance.json),
and summarizes the bone (channel 7) and tendon (channel 6) probability maps
restricted to the pixels the wound-mask already restricted them to (tissue
channels are zero everywhere outside the exact wound mask, not just outside
the wound's bounding box -- see exp3's generate_dataset.py -- so "mean/max
within the wound region" is just "mean/max over the nonzero pixels", no new
masking logic needed here).
"""
import csv
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent          # experiment_extended/wp4_stage34_error_audit/scripts/
WP_DIR = THIS_DIR.parent
EXT_DIR = WP_DIR.parent                              # experiment_extended/
REPO_ROOT = THIS_DIR.parents[3]

PREDICTIONS_CSV = REPO_ROOT / "more_experiments/reports/figures/mit_b1_corn_test_explanations/all_predictions.csv"
TISSUE_DATA_ROOT = REPO_ROOT / "datasets/wound-staging-masks-exp-3"
OUT_CSV = EXT_DIR / "stage34_error_cases.csv"

CLASS_ORDER = ["stage_1", "stage_2", "stage_3", "stage_4"]
STAGE3_IDX, STAGE4_IDX = 2, 3

GROUP_BY_TRUE_PRED = {
    (STAGE3_IDX, STAGE3_IDX): "correct_3",
    (STAGE3_IDX, STAGE4_IDX): "error_3to4",
    (STAGE4_IDX, STAGE4_IDX): "correct_4",
    (STAGE4_IDX, STAGE3_IDX): "error_4to3",
}


def tissue_stats(npy_path):
    """Returns (bone_mean, bone_max, tendon_mean, tendon_max, wound_pixels,
    wound_area_fraction), all None for the probability stats if no wound was
    detected at all (every tissue channel zero everywhere -- see exp-3's own
    "no_wound_detected_behavior" convention, not re-derived here, just
    observed on the array)."""
    arr = np.load(npy_path)  # (H, W, 8) uint8
    tendon = arr[:, :, 6].astype(np.float32) / 255.0
    bone = arr[:, :, 7].astype(np.float32) / 255.0
    wound_mask = arr[:, :, 3:8].sum(axis=-1) > 0
    n_wound = int(wound_mask.sum())
    area_fraction = n_wound / wound_mask.size
    if n_wound == 0:
        return None, None, None, None, 0, 0.0
    bone_vals = bone[wound_mask]
    tendon_vals = tendon[wound_mask]
    return (
        float(bone_vals.mean()), float(bone_vals.max()),
        float(tendon_vals.mean()), float(tendon_vals.max()),
        n_wound, area_fraction,
    )


def main():
    if not PREDICTIONS_CSV.is_file():
        raise SystemExit(f"BLOCKED: {PREDICTIONS_CSV} not found -- expected task_3/explain_test_folds.py's "
                          f"already-generated output. Re-run that script first if it's missing.")

    rows_out = []
    n_skipped_no_npy = 0
    with open(PREDICTIONS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            true_idx = int(row["true_label"])
            pred_idx = int(row["pred_label"])
            group = GROUP_BY_TRUE_PRED.get((true_idx, pred_idx))
            if group is None:
                continue  # not a stage_3/stage_4 case, or a cross-stage mistake outside this WP's scope

            rel_path = row["path"]
            npy_path = TISSUE_DATA_ROOT / Path(rel_path).with_suffix(".npy")
            if not npy_path.is_file():
                n_skipped_no_npy += 1
                print(f"WARNING: no tissue array for {rel_path} ({npy_path}), skipping", file=sys.stderr)
                continue

            bone_mean, bone_max, tendon_mean, tendon_max, n_wound_px, wound_frac = tissue_stats(npy_path)

            rows_out.append({
                "path": rel_path,
                "fold": row["fold"],
                "true_stage": CLASS_ORDER[true_idx],
                "pred_stage": CLASS_ORDER[pred_idx],
                "group": group,
                "predicted_probability": row["predicted_probability"],
                "true_probability": row["true_probability"],
                "max_probability": row["max_probability"],
                "entropy": row["entropy"],
                "prob_argmax": row["prob_argmax"],
                "median_mode_disagree": row["median_mode_disagree"],
                "wound_detected": n_wound_px > 0,
                "wound_area_fraction": round(wound_frac, 4),
                "bone_mean": round(bone_mean, 4) if bone_mean is not None else "",
                "bone_max": round(bone_max, 4) if bone_max is not None else "",
                "tendon_mean": round(tendon_mean, 4) if tendon_mean is not None else "",
                "tendon_max": round(tendon_max, 4) if tendon_max is not None else "",
            })

    rows_out.sort(key=lambda r: (r["group"], r["fold"], r["path"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    from collections import Counter
    counts = Counter(r["group"] for r in rows_out)
    print(f"wrote {OUT_CSV} ({len(rows_out)} rows, {n_skipped_no_npy} skipped for missing tissue array)")
    for group in ("error_3to4", "error_4to3", "correct_3", "correct_4"):
        print(f"  {group}: {counts.get(group, 0)}")


if __name__ == "__main__":
    main()
