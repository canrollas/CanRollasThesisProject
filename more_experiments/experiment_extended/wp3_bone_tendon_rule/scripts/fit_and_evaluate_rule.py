#!/usr/bin/env python3
"""Work Package 3 (advisor extended experiments): rule-based bone/tendon
check for Stage 3 vs 4.

"Contested" = the RGB baseline's OWN decoded CORN prediction is stage_3 or
stage_4 (pred_label in {2,3}) -- this is the literal "model choosing between
Stage 3 and Stage 4" case the brief names, independent of the true label (a
true stage_2 case the model mistakenly leans toward stage_3/4 is still a
case where the model is making that specific choice).

Threshold fitting uses VALIDATION-split predictions ONLY (freshly computed
here -- all_predictions.csv only has the test split). Final before/after
numbers are reported on the TEST split's contested cases only, which the
threshold never saw.

Rule (deliberately asymmetric, per the brief's own caution -- "do not assume
absence of bone/tendon proves Stage 3"): bone/tendon EVIDENCE can push a
stage_3 call up to stage_4; absence of evidence never pushes a stage_4 call
down to stage_3. Concretely:

    if decoded == stage_3 and max(bone_max, tendon_max) >= tau:
        rule_decoded = stage_4
    else:
        rule_decoded = decoded   # stage_4 calls, and any stage_3 call below tau, are left alone

tau is swept over the pooled validation contested cases and chosen to
maximize accuracy restricted to the true stage_3/stage_4 pair (the only
pair this rule can ever change) -- ties broken toward the larger tau (more
conservative, flips fewer cases).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent             # experiment_extended/wp3_bone_tendon_rule/scripts/
WP_DIR = THIS_DIR.parent
EXT_DIR = WP_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]

sys.path.insert(0, str(REPO_ROOT / "task_3"))
from src.dataset import CLASS_ORDER, StageDataset  # noqa: E402
from src.model import build_model as build_rgb_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402
from corn_explain import corn_outputs  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "more_experiments/experiment_extended/wp4_stage34_error_audit/scripts"))
from build_case_table import tissue_stats  # noqa: E402

EXP4_CFG = load_yaml(REPO_ROOT / "more_experiments/exp/exp4_late_fusion/configs/base.yaml")
RGB_CKPT_TEMPLATE = REPO_ROOT / "task_3/results/task_3_outputs/checkpoints/mit_b1__corn_fold{fold}.pt"
STAGE_DATA_ROOT = REPO_ROOT / "datasets/wound-stage-classification"
TISSUE_DATA_ROOT = REPO_ROOT / "datasets/wound-staging-masks-exp-3"
ALL_PREDICTIONS_CSV = REPO_ROOT / "more_experiments/reports/figures/mit_b1_corn_test_explanations/all_predictions.csv"

STAGE3_IDX, STAGE4_IDX = 2, 3

OUT_VAL_ROWS = WP_DIR / "results" / "val_contested_cases.csv"
OUT_TEST_ROWS = WP_DIR / "results" / "test_contested_before_after.csv"
OUT_SUMMARY = WP_DIR / "results" / "rule_summary.csv"


@torch.no_grad()
def predict_val_split(fold_idx, device):
    """Runs the RGB baseline on fold_idx's own VAL split (never used for
    threshold-free test reporting elsewhere), returns per-sample decoded
    label, true label, path, and stage_3/stage_4 class probabilities."""
    rgb_ckpt = Path(str(RGB_CKPT_TEMPLATE).format(fold=fold_idx))
    if not rgb_ckpt.is_file():
        raise SystemExit(f"BLOCKED: {rgb_ckpt} not found.")
    model = build_rgb_model("mit_b1", None, "corn", num_classes=len(CLASS_ORDER))
    model.load_state_dict(torch.load(rgb_ckpt, map_location=device))
    model.to(device).eval()

    folds = build_folds(STAGE_DATA_ROOT, EXP4_CFG["num_folds"], EXP4_CFG["val_fraction"], EXP4_CFG["seed"])
    val_samples = folds[fold_idx]["val"]
    ds = StageDataset(val_samples, STAGE_DATA_ROOT, EXP4_CFG["image_size"], transform=None)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    rows = []
    sample_idx = 0
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        out = corn_outputs(logits)
        decoded = out["decoded"]
        probs = out["class_probabilities"]
        for i in range(len(targets)):
            rel_path, true_idx = val_samples[sample_idx]
            rows.append({
                "path": rel_path, "fold": fold_idx, "true": int(true_idx),
                "decoded": int(decoded[i]),
                "prob_stage3": float(probs[i, STAGE3_IDX]), "prob_stage4": float(probs[i, STAGE4_IDX]),
            })
            sample_idx += 1
    del model
    return rows


def attach_tissue_stats(rows):
    for row in rows:
        npy_path = TISSUE_DATA_ROOT / Path(row["path"]).with_suffix(".npy")
        bone_mean, bone_max, tendon_mean, tendon_max, n_wound, _ = tissue_stats(npy_path)
        row["bone_max"] = bone_max or 0.0
        row["tendon_max"] = tendon_max or 0.0
        row["max_evidence"] = max(row["bone_max"], row["tendon_max"])
    return rows


def apply_rule(decoded, max_evidence, tau):
    if decoded == STAGE3_IDX and max_evidence >= tau:
        return STAGE4_IDX
    return decoded


def sweep_tau(contested_rows):
    """Contested = decoded in {stage_3, stage_4}. Sweeps tau over observed
    max_evidence values, scores accuracy restricted to true in
    {stage_3, stage_4} (the only pair the rule can affect), returns the
    best tau (largest tau on ties -- the more conservative choice)."""
    pair_rows = [r for r in contested_rows if r["true"] in (STAGE3_IDX, STAGE4_IDX)]
    if not pair_rows:
        return 1.0, 0.0  # no val signal at all -- rule never fires (tau above any observed max_evidence)
    candidate_taus = sorted({r["max_evidence"] for r in contested_rows} | {1.0})
    best_tau, best_acc = 1.0, -1.0
    for tau in candidate_taus:
        correct = sum(1 for r in pair_rows if apply_rule(r["decoded"], r["max_evidence"], tau) == r["true"])
        acc = correct / len(pair_rows)
        if acc >= best_acc:  # >= : later (larger) taus in the sorted sweep win ties
            best_acc, best_tau = acc, tau
    return best_tau, best_acc


def main():
    device = get_device()

    print("fitting tau on validation-split contested cases (decoded in {stage_3,stage_4})...")
    val_rows = []
    for fold_idx in range(3):
        val_rows.extend(predict_val_split(fold_idx, device))
    val_rows = attach_tissue_stats(val_rows)
    val_contested = [r for r in val_rows if r["decoded"] in (STAGE3_IDX, STAGE4_IDX)]

    OUT_VAL_ROWS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_VAL_ROWS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_contested[0].keys()) if val_contested else [])
        if val_contested:
            writer.writeheader()
            writer.writerows(val_contested)
    print(f"wrote {OUT_VAL_ROWS} ({len(val_contested)} val contested rows out of {len(val_rows)} val rows total)")

    tau, val_acc = sweep_tau(val_contested)
    print(f"selected tau={tau:.4f} (val contested stage_3/stage_4 accuracy with rule applied: {val_acc:.4f}, "
          f"n={sum(1 for r in val_contested if r['true'] in (STAGE3_IDX, STAGE4_IDX))})")

    print("\napplying tau to TEST-split contested cases (already-computed baseline predictions, "
          "see task_3/explain_test_folds.py's all_predictions.csv)...")
    test_rows = []
    with open(ALL_PREDICTIONS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            pred_idx = int(row["pred_label"])
            if pred_idx not in (STAGE3_IDX, STAGE4_IDX):
                continue
            test_rows.append({
                "path": row["path"], "fold": row["fold"], "true": int(row["true_label"]),
                "decoded": pred_idx,
            })
    test_rows = attach_tissue_stats(test_rows)
    for row in test_rows:
        row["rule_decoded"] = apply_rule(row["decoded"], row["max_evidence"], tau)
        row["true_stage"] = CLASS_ORDER[row["true"]]
        row["before_stage"] = CLASS_ORDER[row["decoded"]]
        row["after_stage"] = CLASS_ORDER[row["rule_decoded"]]
        row["changed"] = row["decoded"] != row["rule_decoded"]

    with open(OUT_TEST_ROWS, "w", newline="") as f:
        fieldnames = ["path", "fold", "true_stage", "before_stage", "after_stage", "changed",
                      "bone_max", "tendon_max", "max_evidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in test_rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"wrote {OUT_TEST_ROWS} ({len(test_rows)} test contested rows)")

    # Before/after, restricted to the true stage_3/stage_4 pair (the only
    # pair this rule can ever touch) -- corrected vs. damaged counts.
    pair_rows = [r for r in test_rows if r["true"] in (STAGE3_IDX, STAGE4_IDX)]
    n_changed = sum(1 for r in pair_rows if r["changed"])
    corrected_3to4 = sum(1 for r in pair_rows if r["true"] == STAGE4_IDX and r["decoded"] == STAGE3_IDX
                          and r["rule_decoded"] == STAGE4_IDX)
    damaged_3 = sum(1 for r in pair_rows if r["true"] == STAGE3_IDX and r["decoded"] == STAGE3_IDX
                     and r["rule_decoded"] == STAGE4_IDX)
    before_acc = sum(1 for r in pair_rows if r["decoded"] == r["true"]) / len(pair_rows)
    after_acc = sum(1 for r in pair_rows if r["rule_decoded"] == r["true"]) / len(pair_rows)
    before_4to3_errors = sum(1 for r in pair_rows if r["true"] == STAGE3_IDX and r["decoded"] == STAGE4_IDX)
    after_4to3_errors = sum(1 for r in pair_rows if r["true"] == STAGE3_IDX and r["rule_decoded"] == STAGE4_IDX)
    before_3to4_errors = sum(1 for r in pair_rows if r["true"] == STAGE4_IDX and r["decoded"] == STAGE3_IDX)
    after_3to4_errors = sum(1 for r in pair_rows if r["true"] == STAGE4_IDX and r["rule_decoded"] == STAGE3_IDX)

    summary = [{
        "tau": round(tau, 4), "val_contested_n": len(val_contested), "val_acc_with_rule": round(val_acc, 4),
        "test_pair_n": len(pair_rows), "n_changed_by_rule": n_changed,
        "before_pair_accuracy": round(before_acc, 4), "after_pair_accuracy": round(after_acc, 4),
        "corrected_stage4_predicted_as_stage3_now_fixed": corrected_3to4,
        "previously_correct_stage3_now_wrongly_flipped_to_stage4": damaged_3,
        "before_true3_pred4_errors": before_3to4_errors, "after_true3_pred4_errors": after_3to4_errors,
        "before_true4_pred3_errors": before_4to3_errors, "after_true4_pred3_errors": after_4to3_errors,
    }]
    with open(OUT_SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {OUT_SUMMARY}")
    print(summary[0])


if __name__ == "__main__":
    main()
