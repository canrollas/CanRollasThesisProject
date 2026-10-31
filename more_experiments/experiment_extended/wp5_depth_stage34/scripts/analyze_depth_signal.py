#!/usr/bin/env python3
"""Work Package 5 (advisor extended experiments): depth cue for Stage 3 vs 4.

Reuses the existing, complete Depth-Anything-V2-Small relative-depth maps at
datasets/wound-stage-depth-anything-v2-small/<path>.npy (raw float32 relative
inverse depth, all 1091 images, same 224x224 resolution as everything else in
this pipeline -- see that dataset's own provenance.json: "image-derived
geometric prior; no metric or measured wound depth"). No depth model is run
here, no new dataset is generated -- these outputs already existed on disk
before this work package started.

(Also found datasets/wound-stage-depth-manual/ -- a 3-image, stage_3-only,
unreviewed manual depth-correction tool output. Not used: no stage_4
coverage at all, so it cannot support a stage_3-vs-4 comparison, and it is
explicitly marked "reviewed": false in its own metadata.)

Step 1 (exploratory, per the brief): for the same stage_3/stage_4 TEST cases
WP4 already tabulated, do simple within-wound depth statistics differ
between correct and incorrect predictions? The wound-region mask reused here
is the SAME nonzero mask the tissue channels already use (exp-3's 8-channel
array), not a new region-detection step.

Step 2: only if step 1 shows a usable signal, extend WP3's rule with the
smallest possible addition (one more threshold on a depth statistic) and
re-report the same before/after numbers WP3 used. If step 1 shows nothing
usable, this script stops after step 1 and reports that as the finding --
a clean negative result is an acceptable, expected outcome per the brief.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

THIS_DIR = Path(__file__).resolve().parent            # experiment_extended/wp5_depth_stage34/scripts/
WP_DIR = THIS_DIR.parent
EXT_DIR = WP_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]

CASES_CSV = EXT_DIR / "stage34_error_cases.csv"
TISSUE_DATA_ROOT = REPO_ROOT / "datasets/wound-staging-masks-exp-3"
DEPTH_DATA_ROOT = REPO_ROOT / "datasets/wound-stage-depth-anything-v2-small"
OUT_CSV = WP_DIR / "results" / "depth_stage34_stats.csv"
OUT_SUMMARY = WP_DIR / "results" / "depth_signal_summary.csv"


def wound_mask_and_depth(rel_path):
    tissue_path = TISSUE_DATA_ROOT / Path(rel_path).with_suffix(".npy")
    depth_path = DEPTH_DATA_ROOT / Path(rel_path).with_suffix(".npy")
    if not tissue_path.is_file() or not depth_path.is_file():
        return None
    tissue_arr = np.load(tissue_path)
    wound_mask = tissue_arr[:, :, 3:8].sum(axis=-1) > 0
    depth = np.load(depth_path)
    if depth.shape != wound_mask.shape:
        return None
    return wound_mask, depth


def depth_stats(rel_path):
    result = wound_mask_and_depth(rel_path)
    if result is None or result[0].sum() == 0:
        return None
    wound_mask, depth = result
    in_wound = depth[wound_mask]
    outside = depth[~wound_mask]
    return {
        "wound_depth_mean": float(in_wound.mean()),
        "wound_depth_std": float(in_wound.std()),
        "wound_depth_range": float(in_wound.max() - in_wound.min()),
        "wound_minus_skin_mean": float(in_wound.mean() - outside.mean()) if outside.size else float("nan"),
    }


def main():
    if not CASES_CSV.is_file():
        raise SystemExit(f"BLOCKED: {CASES_CSV} not found -- run wp4's build_case_table.py first "
                          f"(this script reuses its stage_3/stage_4 case list).")

    with open(CASES_CSV, newline="") as f:
        cases = list(csv.DictReader(f))

    rows = []
    n_skipped = 0
    for row in cases:
        stats = depth_stats(row["path"])
        if stats is None:
            n_skipped += 1
            continue
        rows.append({
            "path": row["path"], "fold": row["fold"], "group": row["group"],
            "true_stage": row["true_stage"], "pred_stage": row["pred_stage"],
            "correct": row["group"] in ("correct_3", "correct_4"),
            **stats,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows, {n_skipped} skipped: missing depth file or empty wound mask)")

    # Step 1: correct vs incorrect, within each true stage, for each stat --
    # exploratory only, reported with a two-sample t-test p-value (NOT a
    # claim of significance after correction for the number of stats tested;
    # flagged as exploratory in summary.md regardless of these p-values).
    stat_names = ["wound_depth_mean", "wound_depth_std", "wound_depth_range", "wound_minus_skin_mean"]
    summary_rows = []
    for true_stage in ("stage_3", "stage_4"):
        subset = [r for r in rows if r["true_stage"] == true_stage]
        correct_vals = {s: [r[s] for r in subset if r["correct"]] for s in stat_names}
        wrong_vals = {s: [r[s] for r in subset if not r["correct"]] for s in stat_names}
        for s in stat_names:
            c, w = np.array(correct_vals[s]), np.array(wrong_vals[s])
            if len(c) < 3 or len(w) < 3:
                continue
            t_stat, p_val = scipy_stats.ttest_ind(c, w, equal_var=False)
            summary_rows.append({
                "true_stage": true_stage, "stat": s,
                "n_correct": len(c), "n_wrong": len(w),
                "mean_correct": round(float(c.mean()), 4), "mean_wrong": round(float(w.mean()), 4),
                "t_stat": round(float(t_stat), 3), "p_value": round(float(p_val), 4),
            })

    with open(OUT_SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {OUT_SUMMARY}")
    for row in summary_rows:
        flag = "  <-- p<0.05 (exploratory, uncorrected)" if row["p_value"] < 0.05 else ""
        print(f"  {row}{flag}")


if __name__ == "__main__":
    main()
