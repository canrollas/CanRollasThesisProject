"""Confusion-aware CORN training losses (Exp-12).

Reuses task_3/corn_explain.py's corn_outputs() for differentiable CORN
class-probability reconstruction instead of reimplementing it -- that
function is already the project's one existing, trusted implementation of
survival-function decoding (used by the held-out explanation pipeline at
more_experiments/reports/figures/mit_b1_corn_test_explanations/, which
motivated this experiment in the first place), so building a second,
independent reconstruction here would risk it silently drifting from the
convention the rest of the thesis already relies on.

All three loss variants (A/B/C below) are auxiliary terms ADDED to the
unchanged base CORN loss (task_3/src/losses.py's CornLoss). None of them
touch the CORN head, its logits, or the decoding rule used for predicted
labels (coral_pytorch's corn_label_from_logits) -- only the training
objective changes. See CombinedLoss for how a variant is selected.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent  # exp/exp12_confusion_aware_loss/scripts/
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from corn_explain import corn_outputs  # noqa: E402
from src.losses import CornLoss  # noqa: E402


def corn_class_probabilities(logits):
    """[B, num_classes-1] raw CORN logits -> ([B, num_classes] differentiable
    class probabilities p_1..p_C, [B, num_classes-1] conditional
    probabilities q_1..q_{C-1}). Thin pass-through to corn_outputs() -- see
    its own docstring for the log-space survival-difference derivation that
    is mathematically identical to the brief's
    p_1=1-q_1, p_2=q_1(1-q_2), p_3=q_1q_2(1-q_3), p_4=q_1q_2q_3 formulas
    (verified directly in test_losses.py against that exact closed form,
    not assumed from the shared implementation alone)."""
    out = corn_outputs(logits)
    return out["class_probabilities"], out["conditional_probabilities"]


def ordinal_distance_loss(class_probabilities, targets):
    """Exp-12A. L_dist = mean_i sum_j p_{i,j} * |y_i - j| / (C-1) -- the
    expected normalized ordinal distance between the true class and the
    model's own (soft) class distribution. Uses no confusion statistics;
    this is the control for Exp-12B's confusion-aware proposal."""
    num_classes = class_probabilities.shape[1]
    classes = torch.arange(num_classes, device=class_probabilities.device, dtype=class_probabilities.dtype)
    target_f = targets.to(class_probabilities.dtype)
    distance = (classes.unsqueeze(0) - target_f.unsqueeze(1)).abs() / (num_classes - 1)
    return (class_probabilities * distance).sum(dim=1).mean()


def build_cost_matrix(confusion_counts, alpha, num_classes=4):
    """Exp-12B. confusion_counts: [C,C] array/tensor of TRAIN-SIDE
    VALIDATION true(rows)-vs-predicted(cols) counts from the already-trained
    baseline checkpoint (see build_cost_matrices.py) -- never the outer test
    fold. Returns a frozen [C,C] float32 tensor:

        C[a,b] = 0                                           if a == b
        C[a,b] = (|a-b|/(C-1)) * (1 + alpha * E[a,b]/E_max)   if a != b

    where E[a,b] = confusion_counts[a,b] / sum_j confusion_counts[a,j]
    (row-normalized, i.e. conditioned on the true class) and E_max is the
    largest off-diagonal entry of E.

    If every off-diagonal confusion_counts entry is zero (E_max == 0 -- a
    validation split with zero observed errors), alpha has no well-defined
    effect: this falls back to the pure ordinal-distance matrix exactly as
    the brief specifies ("use the ordinal-distance matrix without the
    confusion adjustment"), not to alpha=0 silently substituted in, which
    would be a different (if numerically similar) fallback rule."""
    counts = torch.as_tensor(confusion_counts, dtype=torch.float64)
    if counts.shape != (num_classes, num_classes):
        raise ValueError(f"confusion_counts must be [{num_classes},{num_classes}], got {tuple(counts.shape)}")
    if (counts < 0).any():
        raise ValueError("confusion_counts must be non-negative")

    row_sums = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    E = counts / row_sums

    classes = torch.arange(num_classes, dtype=torch.float64)
    ordinal = (classes.unsqueeze(0) - classes.unsqueeze(1)).abs() / (num_classes - 1)  # |a-b|/(C-1)

    off_diag_mask = ~torch.eye(num_classes, dtype=torch.bool)
    e_max = E[off_diag_mask].max() if off_diag_mask.any() else torch.tensor(0.0, dtype=torch.float64)

    if float(e_max) <= 0.0:
        cost = ordinal.clone()
    else:
        cost = ordinal * (1.0 + alpha * E / e_max)

    cost.fill_diagonal_(0.0)
    return cost.to(torch.float32)


def confusion_aware_loss(class_probabilities, targets, cost_matrix):
    """Exp-12B. L_conf = mean_i sum_j C[y_i,j] * p_{i,j}. cost_matrix must
    already be frozen/precomputed by build_cost_matrix() -- this function
    never fits it from the current batch."""
    costs = cost_matrix.to(device=class_probabilities.device, dtype=class_probabilities.dtype)[targets]  # [B, C]
    return (class_probabilities * costs).sum(dim=1).mean()


def boundary_threshold_loss(logits, targets, boundary_index):
    """Exp-12C. L_3: mean binary cross-entropy for ONE CORN conditional
    threshold (0-indexed boundary_index into the [B, num_classes-1] logits;
    boundary_index=2 is the Stage-3/4 boundary for 4 classes), restricted to
    its eligible samples -- the SAME eligibility rule coral_pytorch's own
    corn_loss uses for this task index (true label > boundary_index - 1),
    reproduced here for exactly one threshold instead of summing over all
    of them (see coral_pytorch/losses.py's corn_loss for the reference
    masking this mirrors). Returns a differentiable 0 (not NaN, not a
    detached Python float) if a batch happens to contain no eligible
    sample, so it stays safe inside a larger backward pass."""
    eligible = targets > (boundary_index - 1)
    if not eligible.any():
        return logits.sum() * 0.0
    pred = logits[eligible, boundary_index]
    label = (targets[eligible] > boundary_index).to(pred.dtype)
    per_sample = -(F.logsigmoid(pred) * label + (F.logsigmoid(pred) - pred) * (1 - label))
    return per_sample.mean()


class CombinedLoss(nn.Module):
    """base CornLoss (task_3/src/losses.py, UNCHANGED) + one auxiliary term,
    selected by `variant`:

      'baseline' -- exactly the original CornLoss, no auxiliary term added
                    at all (not an auxiliary term with weight 0 -- so the
                    brief's "baseline should also be reproducible under its
                    original loss" is an EXACT reproduction, verified in
                    test_losses.py via torch.equal against CornLoss alone).
      'A' -- + lam * ordinal_distance_loss(...)           (Exp-12A)
      'B' -- + lam * confusion_aware_loss(..., cost_matrix) (Exp-12B)
      'C' -- + beta * boundary_threshold_loss(...)         (Exp-12C)

    cost_matrix (variant 'B' only) is registered as a buffer so it moves
    with .to(device) automatically and is never treated as a trainable
    parameter.
    """

    VARIANTS = ("baseline", "A", "B", "C")

    def __init__(self, num_classes, variant, lam=0.0, cost_matrix=None, beta=0.0, boundary_index=None):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant {variant!r}, expected one of {self.VARIANTS}")
        if variant == "B" and cost_matrix is None:
            raise ValueError("variant 'B' requires a precomputed cost_matrix")
        self.num_classes = num_classes
        self.variant = variant
        self.lam = lam
        self.beta = beta
        self.boundary_index = boundary_index if boundary_index is not None else num_classes - 2
        self.base = CornLoss(num_classes)
        if cost_matrix is not None:
            self.register_buffer("cost_matrix", cost_matrix)
        else:
            self.cost_matrix = None

    def forward(self, logits, targets):
        base_loss = self.base(logits, targets)
        if self.variant == "baseline":
            return base_loss
        if self.variant == "A":
            class_probs, _ = corn_class_probabilities(logits)
            return base_loss + self.lam * ordinal_distance_loss(class_probs, targets)
        if self.variant == "B":
            class_probs, _ = corn_class_probabilities(logits)
            return base_loss + self.lam * confusion_aware_loss(class_probs, targets, self.cost_matrix)
        if self.variant == "C":
            return base_loss + self.beta * boundary_threshold_loss(logits, targets, self.boundary_index)
        raise AssertionError("unreachable: variant already validated in __init__")
