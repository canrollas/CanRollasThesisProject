#!/usr/bin/env python3
"""Oracle-complementarity analysis for exp-11's CORN vs. stage3-probe
predictions. NOT a training experiment -- no new model is fit here, no
threshold is tuned on anything. Pure post-hoc analysis of predictions
and confidences already saved by extract_features.py / train_probe.py.

Two questions, both read-only over existing artifacts:

1. Oracle upper bound: if a perfect selector always picked whichever of
   {CORN, stage3-probe} was correct, what test accuracy would that reach
   (pooled over all 3 folds)? This is `1 - both_wrong_rate`, NOT a
   claim that any real selector can achieve it.

2. On the subset where CORN and the stage3-probe DISAGREE (the only
   subset where a selector could possibly change anything), are their
   confidences actually informative about which one is right? CORN's
   confidence is derived from its own ordinal conditional logits
   (recomputed from the already-saved 512-dim final_features and the
   checkpoint's own CornHead weights -- no re-extraction, no encoder
   forward pass); the stage3-probe's confidence is its own
   `predict_proba` output (train_probe.py now saves this).

   A single, PARAMETER-FREE heuristic ("trust whichever model reports
   higher confidence") is evaluated directly on the disagreement subset
   as a diagnostic -- this has zero degrees of freedom tuned on test
   data (no threshold, no fitted weights), unlike an actual selector
   model, which per the proposal's own warning would need to be chosen
   on train/val, never on these same 3 test folds.
"""
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import CornHead  # noqa: E402
from src.utils import load_yaml  # noqa: E402

FOLDS = (0, 1, 2)


def corn_class_probs(logits_np, num_classes):
    """Raw conditional CORN logits [N, K-1] -> per-class probabilities
    [N, K], via the standard rank-consistent-model decomposition
    P(y=0)=1-P(y>0), P(y=k)=P(y>k-1)-P(y>k), P(y=K-1)=P(y>K-2).
    Independently-trained sigmoids aren't guaranteed perfectly monotonic
    for every sample, so negative entries are clipped to 0 and the row
    is renormalized -- a confidence SCORE for the already-decided label,
    not a re-decision of the label itself (that stays `test_corn_preds`,
    from `coral_pytorch`'s own decode rule)."""
    p_gt = torch.sigmoid(torch.from_numpy(logits_np)).numpy()  # [N, K-1], P(y>k)
    probs = np.zeros((logits_np.shape[0], num_classes))
    probs[:, 0] = 1 - p_gt[:, 0]
    for k in range(1, num_classes - 1):
        probs[:, k] = p_gt[:, k - 1] - p_gt[:, k]
    probs[:, num_classes - 1] = p_gt[:, num_classes - 2]
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def load_corn_head(checkpoint_path, in_dim, num_classes):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    head_state = {k[len("1."):]: v for k, v in state_dict.items() if k.startswith("1.")}
    head = CornHead(in_dim, num_classes)
    head.load_state_dict(head_state, strict=True)
    head.eval()
    return head


def per_fold_data(fold, base_cfg, num_classes):
    fold_dir = EXP_DIR / "results" / f"fold_{fold}"
    labels = np.load(fold_dir / "test_labels.npy")
    corn_preds = np.load(fold_dir / "test_corn_preds.npy")
    final_features = np.load(fold_dir / "test_final_features.npy")
    stage3_preds = np.load(fold_dir / "probe_stage3_test_preds.npy")
    stage3_proba = np.load(fold_dir / "probe_stage3_test_proba.npy")
    stage3_classes = np.load(fold_dir / "probe_stage3_classes.npy")

    checkpoint_path = REPO_ROOT / base_cfg["rgb_checkpoint_template"].format(fold=fold)
    head = load_corn_head(checkpoint_path, final_features.shape[1], num_classes)
    with torch.no_grad():
        corn_logits = head(torch.from_numpy(final_features)).numpy()
    corn_probs = corn_class_probs(corn_logits, num_classes)
    corn_confidence = corn_probs[np.arange(len(labels)), corn_preds]

    class_to_col = {c: i for i, c in enumerate(stage3_classes)}
    stage3_col = np.array([class_to_col[p] for p in stage3_preds])
    stage3_confidence = stage3_proba[np.arange(len(labels)), stage3_col]

    return {
        "labels": labels, "corn_preds": corn_preds, "stage3_preds": stage3_preds,
        "corn_confidence": corn_confidence, "stage3_confidence": stage3_confidence,
    }


def main():
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    num_classes = len(base_cfg["classes"])

    pooled_n = 0
    pooled_oracle_correct = 0
    pooled_corn_correct = 0
    pooled_stage3_correct = 0
    pooled_both_wrong = 0
    pooled_disagree = 0
    pooled_disagree_corn_only = 0
    pooled_disagree_stage3_only = 0
    pooled_disagree_both_wrong = 0
    pooled_selector_correct = 0
    conf_corn_when_corn_only_right, conf_stage3_when_corn_only_right = [], []
    conf_corn_when_stage3_only_right, conf_stage3_when_stage3_only_right = [], []
    conf_corn_when_both_wrong, conf_stage3_when_both_wrong = [], []

    print(f"{'fold':<5}{'n':<6}{'corn_acc':<10}{'stage3_acc':<12}{'oracle_acc':<12}{'n_disagree':<12}")
    for fold in FOLDS:
        d = per_fold_data(fold, base_cfg, num_classes)
        n = len(d["labels"])
        corn_correct = d["corn_preds"] == d["labels"]
        stage3_correct = d["stage3_preds"] == d["labels"]
        oracle_correct = corn_correct | stage3_correct
        both_wrong = ~corn_correct & ~stage3_correct
        disagree = d["corn_preds"] != d["stage3_preds"]

        assert not np.any(corn_correct & stage3_correct & disagree), \
            "disagreeing predictions can't both equal the true label -- decode or join bug"

        disagree_corn_only = disagree & corn_correct & ~stage3_correct
        disagree_stage3_only = disagree & ~corn_correct & stage3_correct
        disagree_both_wrong = disagree & ~corn_correct & ~stage3_correct

        selector_pred_is_corn = d["corn_confidence"] >= d["stage3_confidence"]
        selector_correct_in_disagree = np.where(
            selector_pred_is_corn[disagree], corn_correct[disagree], stage3_correct[disagree]
        )

        print(f"{fold:<5}{n:<6}{corn_correct.mean():<10.4f}{stage3_correct.mean():<12.4f}"
              f"{oracle_correct.mean():<12.4f}{int(disagree.sum()):<12}")

        pooled_n += n
        pooled_oracle_correct += int(oracle_correct.sum())
        pooled_corn_correct += int(corn_correct.sum())
        pooled_stage3_correct += int(stage3_correct.sum())
        pooled_both_wrong += int(both_wrong.sum())
        pooled_disagree += int(disagree.sum())
        pooled_disagree_corn_only += int(disagree_corn_only.sum())
        pooled_disagree_stage3_only += int(disagree_stage3_only.sum())
        pooled_disagree_both_wrong += int(disagree_both_wrong.sum())
        pooled_selector_correct += int(selector_correct_in_disagree.sum())

        conf_corn_when_corn_only_right.extend(d["corn_confidence"][disagree_corn_only])
        conf_stage3_when_corn_only_right.extend(d["stage3_confidence"][disagree_corn_only])
        conf_corn_when_stage3_only_right.extend(d["corn_confidence"][disagree_stage3_only])
        conf_stage3_when_stage3_only_right.extend(d["stage3_confidence"][disagree_stage3_only])
        conf_corn_when_both_wrong.extend(d["corn_confidence"][disagree_both_wrong])
        conf_stage3_when_both_wrong.extend(d["stage3_confidence"][disagree_both_wrong])

    print(f"\n=== pooled over {len(FOLDS)} folds, n={pooled_n} ===")
    print(f"CORN accuracy:          {pooled_corn_correct / pooled_n:.4f}")
    print(f"stage3-probe accuracy:  {pooled_stage3_correct / pooled_n:.4f}")
    print(f"ORACLE accuracy:        {pooled_oracle_correct / pooled_n:.4f}  "
          f"(= 1 - both_wrong_rate = 1 - {pooled_both_wrong}/{pooled_n})")

    print(f"\n=== disagreement subset: n={pooled_disagree} "
          f"({pooled_disagree / pooled_n:.1%} of all test images) ===")
    print(f"  CORN-only-correct:     {pooled_disagree_corn_only} "
          f"({pooled_disagree_corn_only / pooled_disagree:.4f})")
    print(f"  stage3-only-correct:   {pooled_disagree_stage3_only} "
          f"({pooled_disagree_stage3_only / pooled_disagree:.4f})")
    print(f"  both wrong:            {pooled_disagree_both_wrong} "
          f"({pooled_disagree_both_wrong / pooled_disagree:.4f})")
    oracle_rate_in_disagree = (pooled_disagree_corn_only + pooled_disagree_stage3_only) / pooled_disagree
    print(f"  oracle accuracy HERE:  {oracle_rate_in_disagree:.4f}  (best any selector could reach on this subset)")
    print(f"  always-trust-CORN:     {pooled_disagree_corn_only / pooled_disagree:.4f}  (= CORN's own accuracy here)")
    print(f"  always-trust-stage3:   {pooled_disagree_stage3_only / pooled_disagree:.4f}")
    print(f"  higher-confidence heuristic (0 free parameters): "
          f"{pooled_selector_correct / pooled_disagree:.4f}")

    def report_conf(name, corn_vals, stage3_vals):
        corn_vals, stage3_vals = np.array(corn_vals), np.array(stage3_vals)
        print(f"  [{name}] n={len(corn_vals)} "
              f"mean_corn_conf={corn_vals.mean():.4f} mean_stage3_conf={stage3_vals.mean():.4f}")

    print("\n=== mean confidence by who's actually correct, within disagreement ===")
    report_conf("CORN-only-correct   ", conf_corn_when_corn_only_right, conf_stage3_when_corn_only_right)
    report_conf("stage3-only-correct ", conf_corn_when_stage3_only_right, conf_stage3_when_stage3_only_right)
    report_conf("both wrong          ", conf_corn_when_both_wrong, conf_stage3_when_both_wrong)


if __name__ == "__main__":
    main()
