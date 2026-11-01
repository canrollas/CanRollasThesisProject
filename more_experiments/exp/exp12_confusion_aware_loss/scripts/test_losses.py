#!/usr/bin/env python3
"""Correctness checks for losses.py, run before trusting any training run
from this experiment. Not a pytest suite -- a short, explicit script
printing PASS/FAIL per check, matching every other exp/*/scripts/test_*.py
in this repo. Run with: python test_losses.py

Checks:
  1. corn_class_probabilities matches the brief's closed-form formula
     (p1=1-q1, p2=q1(1-q2), p3=q1q2(1-q3), p4=q1q2q3) exactly, on a known
     (q1,q2,q3) triple -- not just "sums to 1".
  2. Class probabilities sum to 1 for a batch of random logits.
  3. build_cost_matrix: diagonal is exactly zero.
  4. build_cost_matrix: matches a hand-worked asymmetric confusion example
     exactly (not just "is asymmetric").
  5. build_cost_matrix: falls back to pure ordinal distance when every
     off-diagonal confusion count is zero.
  6. ordinal_distance_loss / confusion_aware_loss / boundary_threshold_loss
     all produce finite gradients w.r.t. the input logits.
  7. boundary_threshold_loss: eligibility mask is exactly
     {targets > boundary_index - 1}, verified against a hand-computed
     2-eligible-sample batch, and returns a safe differentiable zero (not
     NaN) when no sample in the batch is eligible.
  8. CombinedLoss(variant='baseline') is torch.equal to CornLoss alone --
     exact reproduction, not approximate.
  9. CombinedLoss variants A/B/C all produce finite losses and finite,
     non-zero gradients on a small synthetic batch.
"""
import math
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))
sys.path.insert(0, str(THIS_DIR))

from src.losses import CornLoss  # noqa: E402
from losses import (  # noqa: E402
    CombinedLoss,
    boundary_threshold_loss,
    build_cost_matrix,
    confusion_aware_loss,
    corn_class_probabilities,
    ordinal_distance_loss,
)

NUM_CLASSES = 4


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def logit(p):
    return math.log(p / (1 - p))


def main():
    all_ok = True

    # --- 1. closed-form class-probability reconstruction -------------------
    q1, q2, q3 = 0.9, 0.5, 0.2
    logits = torch.tensor([[logit(q1), logit(q2), logit(q3)]])
    class_probs, conditional = corn_class_probabilities(logits)
    expected = torch.tensor([[1 - q1, q1 * (1 - q2), q1 * q2 * (1 - q3), q1 * q2 * q3]])
    all_ok &= check(
        "corn_class_probabilities matches the brief's closed-form formula exactly",
        torch.allclose(class_probs, expected, atol=1e-5),
    )
    all_ok &= check(
        "conditional_probabilities match sigmoid(logits) (q1,q2,q3) exactly",
        torch.allclose(conditional, torch.tensor([[q1, q2, q3]]), atol=1e-5),
    )

    # --- 2. sums to 1 over a random batch -----------------------------------
    torch.manual_seed(0)
    rand_logits = torch.randn(16, NUM_CLASSES - 1) * 3
    rand_probs, _ = corn_class_probabilities(rand_logits)
    all_ok &= check(
        "class probabilities sum to 1 for a batch of random logits",
        torch.allclose(rand_probs.sum(dim=1), torch.ones(16), atol=1e-4),
    )

    # --- 3/4/5. build_cost_matrix -------------------------------------------
    counts = torch.tensor([
        [80., 20., 0., 0.],
        [5., 85., 10., 0.],
        [0., 10., 70., 20.],
        [0., 0., 30., 70.],
    ])
    cost = build_cost_matrix(counts, alpha=1.0, num_classes=NUM_CLASSES)
    all_ok &= check(
        "build_cost_matrix diagonal is exactly zero",
        torch.equal(cost.diagonal(), torch.zeros(NUM_CLASSES)),
    )
    # hand-worked: E_max = E[3,2] = 30/100 = 0.30
    expected_c01 = (1 / 3) * (1 + 1.0 * (20 / 100) / 0.30)
    expected_c32 = (1 / 3) * (1 + 1.0 * (30 / 100) / 0.30)
    expected_c02 = (2 / 3) * (1 + 1.0 * 0.0 / 0.30)  # no observed stage_1->stage_3 confusion
    all_ok &= check(
        "build_cost_matrix matches the hand-worked asymmetric example exactly",
        math.isclose(cost[0, 1].item(), expected_c01, rel_tol=1e-4)
        and math.isclose(cost[3, 2].item(), expected_c32, rel_tol=1e-4)
        and math.isclose(cost[0, 2].item(), expected_c02, rel_tol=1e-4),
    )
    all_ok &= check(
        "build_cost_matrix is asymmetric when the observed confusion is asymmetric",
        not math.isclose(cost[0, 1].item(), cost[1, 0].item(), rel_tol=1e-6),
    )

    zero_counts = torch.tensor([
        [10., 0., 0., 0.],
        [0., 10., 0., 0.],
        [0., 0., 10., 0.],
        [0., 0., 0., 10.],
    ])
    fallback_cost = build_cost_matrix(zero_counts, alpha=1.0, num_classes=NUM_CLASSES)
    ordinal_only = torch.tensor([
        [abs(a - b) / (NUM_CLASSES - 1) for b in range(NUM_CLASSES)] for a in range(NUM_CLASSES)
    ])
    all_ok &= check(
        "build_cost_matrix falls back to pure ordinal distance when there are no observed errors",
        torch.allclose(fallback_cost, ordinal_only, atol=1e-6),
    )

    # --- 6. gradient flow ----------------------------------------------------
    grad_logits = torch.randn(8, NUM_CLASSES - 1, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 0, 2, 3, 1])
    class_probs, _ = corn_class_probabilities(grad_logits)

    l_dist = ordinal_distance_loss(class_probs, targets)
    l_dist.backward(retain_graph=True)
    all_ok &= check(
        "ordinal_distance_loss produces finite, non-zero gradients",
        grad_logits.grad is not None and torch.isfinite(grad_logits.grad).all() and grad_logits.grad.abs().sum() > 0,
    )
    grad_logits.grad = None

    l_conf = confusion_aware_loss(class_probs, targets, cost)
    l_conf.backward(retain_graph=True)
    all_ok &= check(
        "confusion_aware_loss produces finite, non-zero gradients",
        grad_logits.grad is not None and torch.isfinite(grad_logits.grad).all() and grad_logits.grad.abs().sum() > 0,
    )
    grad_logits.grad = None

    l_bound = boundary_threshold_loss(grad_logits, targets, boundary_index=2)
    l_bound.backward()
    all_ok &= check(
        "boundary_threshold_loss produces finite, non-zero gradients",
        grad_logits.grad is not None and torch.isfinite(grad_logits.grad).all() and grad_logits.grad.abs().sum() > 0,
    )

    # --- 7. boundary eligibility mask + empty-batch safety --------------------
    small_logits = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, -0.4], [0.5, 0.1, 0.9], [-0.2, 0.3, 0.7]],
                                 requires_grad=True)
    small_targets = torch.tensor([0, 1, 2, 3])  # only indices 2,3 are eligible for boundary_index=2
    manual_pred = small_logits[[2, 3], 2]
    manual_label = torch.tensor([0.0, 1.0])  # stage_3 -> not-stage_4 (0), stage_4 -> is-stage_4 (1)
    manual_loss = torch.nn.functional.binary_cross_entropy_with_logits(manual_pred, manual_label)
    fn_loss = boundary_threshold_loss(small_logits, small_targets, boundary_index=2)
    all_ok &= check(
        "boundary_threshold_loss eligibility mask matches targets > boundary_index - 1 exactly",
        torch.allclose(fn_loss, manual_loss, atol=1e-5),
    )

    no_eligible_targets = torch.tensor([0, 0, 1, 1])
    safe_loss = boundary_threshold_loss(small_logits, no_eligible_targets, boundary_index=2)
    all_ok &= check(
        "boundary_threshold_loss returns a safe differentiable zero when no sample is eligible",
        torch.isfinite(safe_loss).item() and safe_loss.item() == 0.0,
    )

    # --- 8. baseline variant reproduces CornLoss exactly ----------------------
    base_criterion = CornLoss(NUM_CLASSES)
    combined_baseline = CombinedLoss(NUM_CLASSES, variant="baseline")
    sample_logits = torch.randn(10, NUM_CLASSES - 1)
    sample_targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 1, 2])
    all_ok &= check(
        "CombinedLoss(variant='baseline') is torch.equal to CornLoss alone",
        torch.equal(combined_baseline(sample_logits, sample_targets), base_criterion(sample_logits, sample_targets)),
    )

    # --- 9. variants A/B/C all finite with real gradients ----------------------
    for variant, kwargs in [
        ("A", dict(lam=0.1)),
        ("B", dict(lam=0.1, cost_matrix=cost)),
        ("C", dict(beta=0.5)),
    ]:
        criterion = CombinedLoss(NUM_CLASSES, variant=variant, **kwargs)
        logits_v = torch.randn(10, NUM_CLASSES - 1, requires_grad=True)
        loss_v = criterion(logits_v, sample_targets)
        loss_v.backward()
        ok = (
            torch.isfinite(loss_v).item()
            and logits_v.grad is not None
            and torch.isfinite(logits_v.grad).all()
            and logits_v.grad.abs().sum() > 0
        )
        all_ok &= check(f"CombinedLoss variant='{variant}' produces a finite loss with finite, non-zero gradients", ok)

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
