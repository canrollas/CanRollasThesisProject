#!/usr/bin/env python3
"""Work Package 5, step 2: test the depth cue found in step 1
(analyze_depth_signal.py) as the smallest possible extension to WP3's rule.

Step 1's finding (exploratory, uncorrected p-values, but consistent and
directionally coherent across both true stage_3 and true stage_4 groups):
`wound_minus_skin_mean` (mean relative inverse depth inside the wound minus
mean relative inverse depth on the surrounding skin -- negative means the
wound looks farther from the camera than the skin, i.e. a visual cavity) is
more negative for correctly-classified stage_4 cases and for INcorrectly-
classified stage_3 cases (the ones that get confused toward stage_4) than
for their respective correct/wrong counterparts. In other words: a
cavity-like depth signature co-occurs with "this case looks like stage_4",
whether or not that matches the true label -- exactly the kind of cue WP3's
bone/tendon rule was trying to use, from a different source.

Extension (minimal: one more OR-branch, same tau-on-validation-only
discipline as WP3, same asymmetric direction -- a depth signal can push a
stage_3 call up to stage_4, never the reverse, same reasoning as WP3's
caution about absence-of-evidence):

    if decoded == stage_3 and (max_evidence >= tau_tissue or wound_minus_skin_mean <= tau_depth):
        rule_decoded = stage_4

tau_tissue is carried over from WP3's own fit (1.0 -- i.e. inert, WP3 found
no tissue threshold improved validation accuracy at all), so this script
effectively tests whether depth alone, added on top of that inert tissue
condition, changes anything. tau_depth is fit fresh here on the SAME
validation-split contested cases WP3 used (wp3_bone_tendon_rule/results/
val_contested_cases.csv), with depth stats attached.
"""
import csv
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent             # experiment_extended/wp5_depth_stage34/scripts/
WP_DIR = THIS_DIR.parent
EXT_DIR = WP_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]

sys.path.insert(0, str(THIS_DIR))
from analyze_depth_signal import wound_mask_and_depth  # noqa: E402

WP3_VAL_CSV = EXT_DIR / "wp3_bone_tendon_rule/results/val_contested_cases.csv"
WP3_TEST_CSV = EXT_DIR / "wp3_bone_tendon_rule/results/test_contested_before_after.csv"
TAU_TISSUE = 1.0  # carried over from WP3's own fit -- effectively inert there

STAGE3_IDX, STAGE4_IDX = 2, 3
CLASS_ORDER = ["stage_1", "stage_2", "stage_3", "stage_4"]

OUT_SUMMARY = WP_DIR / "results" / "depth_extended_rule_summary.csv"


def wound_minus_skin_mean(rel_path):
    result = wound_mask_and_depth(rel_path)
    if result is None or result[0].sum() == 0:
        return None
    wound_mask, depth = result
    in_wound = depth[wound_mask]
    outside = depth[~wound_mask]
    if outside.size == 0:
        return None
    return float(in_wound.mean() - outside.mean())


def apply_rule(decoded, max_evidence, wms, tau_tissue, tau_depth):
    if decoded == STAGE3_IDX and (max_evidence >= tau_tissue or (wms is not None and wms <= tau_depth)):
        return STAGE4_IDX
    return decoded


def sweep_tau_depth(val_rows):
    pair_rows = [r for r in val_rows if r["true"] in (STAGE3_IDX, STAGE4_IDX) and r["wms"] is not None]
    if not pair_rows:
        return None, 0.0
    candidates = sorted({r["wms"] for r in pair_rows})
    best_tau, best_acc = None, -1.0
    # ascending tau_depth = rule fires on MORE cases (<=); prefer the
    # SMALLEST tau achieving the best accuracy on ties (fires least, most
    # conservative) -- opposite iteration-order convention from WP3's
    # tie-break (there, larger tau = fires less; here, smaller tau = fires
    # less), same underlying "prefer firing least on ties" principle.
    for tau in candidates:
        correct = sum(
            1 for r in pair_rows
            if apply_rule(r["decoded"], r["max_evidence"], r["wms"], TAU_TISSUE, tau) == r["true"]
        )
        acc = correct / len(pair_rows)
        if acc > best_acc:
            best_acc, best_tau = acc, tau
    if best_tau is None:
        best_tau = -1e9  # never fires
    return best_tau, best_acc


def main():
    if not WP3_VAL_CSV.is_file() or not WP3_TEST_CSV.is_file():
        raise SystemExit("BLOCKED: run wp3_bone_tendon_rule/scripts/fit_and_evaluate_rule.py first.")

    with open(WP3_VAL_CSV, newline="") as f:
        val_rows = []
        for row in csv.DictReader(f):
            val_rows.append({
                "path": row["path"], "true": int(row["true"]), "decoded": int(row["decoded"]),
                "max_evidence": float(row["max_evidence"]),
                "wms": wound_minus_skin_mean(row["path"]),
            })

    tau_depth, val_acc = sweep_tau_depth(val_rows)
    print(f"selected tau_depth={tau_depth}, val pair accuracy with extended rule={val_acc:.4f}")

    with open(WP3_TEST_CSV, newline="") as f:
        test_rows = []
        for row in csv.DictReader(f):
            true_idx = CLASS_ORDER.index(row["true_stage"])
            decoded_idx = CLASS_ORDER.index(row["before_stage"])
            test_rows.append({
                "path": row["path"], "true": true_idx, "decoded": decoded_idx,
                "max_evidence": float(row["max_evidence"]),
                "wms": wound_minus_skin_mean(row["path"]),
            })

    for row in test_rows:
        row["rule_decoded"] = apply_rule(row["decoded"], row["max_evidence"], row["wms"], TAU_TISSUE, tau_depth)

    pair_rows = [r for r in test_rows if r["true"] in (STAGE3_IDX, STAGE4_IDX)]
    before_acc = sum(1 for r in pair_rows if r["decoded"] == r["true"]) / len(pair_rows)
    after_acc = sum(1 for r in pair_rows if r["rule_decoded"] == r["true"]) / len(pair_rows)
    n_changed = sum(1 for r in pair_rows if r["decoded"] != r["rule_decoded"])
    before_3to4 = sum(1 for r in pair_rows if r["true"] == STAGE4_IDX and r["decoded"] == STAGE3_IDX)
    after_3to4 = sum(1 for r in pair_rows if r["true"] == STAGE4_IDX and r["rule_decoded"] == STAGE3_IDX)
    before_4to3 = sum(1 for r in pair_rows if r["true"] == STAGE3_IDX and r["decoded"] == STAGE4_IDX)
    after_4to3 = sum(1 for r in pair_rows if r["true"] == STAGE3_IDX and r["rule_decoded"] == STAGE4_IDX)
    damaged = sum(1 for r in pair_rows if r["true"] == STAGE3_IDX and r["decoded"] == STAGE3_IDX
                   and r["rule_decoded"] == STAGE4_IDX)
    corrected = sum(1 for r in pair_rows if r["true"] == STAGE4_IDX and r["decoded"] == STAGE3_IDX
                     and r["rule_decoded"] == STAGE4_IDX)

    summary = [{
        "tau_depth": round(tau_depth, 4) if tau_depth is not None else "",
        "val_pair_acc_with_extended_rule": round(val_acc, 4),
        "test_pair_n": len(pair_rows), "n_changed_by_rule": n_changed,
        "before_pair_accuracy": round(before_acc, 4), "after_pair_accuracy": round(after_acc, 4),
        "before_true4_pred3_errors": before_3to4, "after_true4_pred3_errors": after_3to4,
        "before_true3_pred4_errors": before_4to3, "after_true3_pred4_errors": after_4to3,
        "corrected_4to3_errors": corrected, "damaged_previously_correct_3": damaged,
    }]
    with open(OUT_SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {OUT_SUMMARY}")
    print(summary[0])


if __name__ == "__main__":
    main()
