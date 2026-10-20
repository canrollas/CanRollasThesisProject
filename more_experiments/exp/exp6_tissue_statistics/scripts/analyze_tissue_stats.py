#!/usr/bin/env python3
"""Step 6: per-stage distribution of the 20 tissue-statistics features --
which ones actually differ across Stage 1-4 (one-way ANOVA F-statistic per
feature, plus per-stage mean+-std), ranked by discriminative power.
Images with zero wound pixels (all-zero feature vector) are excluded --
their "0" isn't a real tissue measurement, including them would just
inflate variance for the stage(s) they happen to fall in.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent

FEATURE_COLUMNS = [
    f"{c}_{s}" for c in ["granulation", "slough", "necrosis", "tendon", "bone"]
    for s in ["mean", "std", "p95", "frac_above_0.5"]
]
STAGES = ["stage_1", "stage_2", "stage_3", "stage_4"]


def main():
    rows = []
    with open(EXP_DIR / "tissue_statistics.csv", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["n_wound_px"]) > 0:  # exclude zero-wound (not-a-measurement) rows
                rows.append(row)

    n_excluded_total = sum(1 for _ in open(EXP_DIR / "tissue_statistics.csv")) - 1 - len(rows)
    print(f"{len(rows)} images with n_wound_px > 0 analyzed ({n_excluded_total} zero-wound images excluded)\n")

    by_stage = {s: [] for s in STAGES}
    for row in rows:
        by_stage[row["stage"]].append(row)
    for s in STAGES:
        print(f"{s}: n={len(by_stage[s])}")
    print()

    results = []
    for feat in FEATURE_COLUMNS:
        groups = [np.array([float(r[feat]) for r in by_stage[s]]) for s in STAGES]
        f_stat, p_value = scipy_stats.f_oneway(*groups)
        means = [g.mean() for g in groups]
        stds = [g.std() for g in groups]
        results.append({"feature": feat, "f_stat": f_stat, "p_value": p_value,
                         "means_by_stage": means, "stds_by_stage": stds})

    results.sort(key=lambda r: r["f_stat"], reverse=True)

    print(f"{'feature':<28} {'F-stat':>8} {'p-value':>10}   mean(stage1 / stage2 / stage3 / stage4)")
    print("-" * 100)
    for r in results:
        means_str = " / ".join(f"{m:.3f}" for m in r["means_by_stage"])
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
        print(f"{r['feature']:<28} {r['f_stat']:>8.2f} {r['p_value']:>10.2e} {sig:<4} {means_str}")

    out_path = EXP_DIR / "tissue_statistics_anova.csv"
    with open(out_path, "w", newline="") as f:
        fieldnames = ["feature", "f_stat", "p_value"] + [f"mean_{s}" for s in STAGES] + [f"std_{s}" for s in STAGES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {"feature": r["feature"], "f_stat": round(r["f_stat"], 4), "p_value": r["p_value"]}
            row.update({f"mean_{s}": round(m, 6) for s, m in zip(STAGES, r["means_by_stage"])})
            row.update({f"std_{s}": round(sd, 6) for s, sd in zip(STAGES, r["stds_by_stage"])})
            writer.writerow(row)
    print(f"\nwrote {out_path}")
    print("\n(*** p<0.001, ** p<0.01, * p<0.05 -- higher F-stat = more separation between stage means "
          "relative to within-stage variance, i.e. more discriminative for staging)")


if __name__ == "__main__":
    main()
