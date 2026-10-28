"""Boundary-specific supervised contrastive loss ("contrastive" mode
only, see model.py's module docstring for the full ablation design).

`boundary_binary_targets` turns the 4-way stage label into the same
three binary splits CORN's own conditional probabilities already
train on (verified against coral_pytorch's corn_loss/corn_label_from_logits
semantics in test_model.py): boundary 0 asks "stage > stage_1?", boundary
1 "stage > stage_2?", boundary 2 "stage > stage_3?" -- exactly the three
questions in the architecture proposal ("Yara Stage k'den ileri mi?").

`BoundarySupConLoss` applies a standard supervised contrastive loss
(Khosla et al., 2020, "Supervised Contrastive Learning") separately to
EACH boundary's own pooled feature z_k against that boundary's own
binary split -- pulling same-side images together and pushing
opposite-side images apart, in z_k's own representation space, never
shared with another boundary's loss term.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def boundary_binary_targets(labels, num_boundaries):
    """labels: [B] long stage indices 0..num_classes-1 (0=stage_1).
    Returns [B, num_boundaries] binary targets: column k is
    1 if `labels > k` else 0."""
    boundaries = torch.arange(num_boundaries, device=labels.device).unsqueeze(0)  # [1, K]
    return (labels.unsqueeze(1) > boundaries).long()  # [B, K]


class BoundarySupConLoss(nn.Module):
    """A boundary's batch-level term is skipped (contributes exactly 0,
    not NaN) if that boundary's binary split happens to have only one
    side present in the current batch -- there is nothing to contrast
    against. With batch_size=16 (same as every other experiment in this
    chapter) this is a real possibility for the more lopsided boundaries
    (e.g. few stage_4 images in a batch for boundary 2), not an edge case
    to assume away.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def _single_boundary_loss(self, z, binary_labels):
        n = z.shape[0]
        if n < 2 or binary_labels.unique().numel() < 2:
            return z.new_zeros(())

        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.t()) / self.temperature  # [n, n]
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(self_mask, float("-inf"))

        same_label = binary_labels.unsqueeze(0) == binary_labels.unsqueeze(1)  # [n, n]
        positive_mask = same_label & ~self_mask
        has_positive = positive_mask.any(dim=1)
        if not bool(has_positive.any()):
            return z.new_zeros(())

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        # log_prob at the masked-out self position is -inf (sim was set to
        # -inf there); torch.where avoids `-inf * 0 = nan` from multiplying
        # it directly by positive_mask (which is always False at self).
        masked_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
        per_anchor = -masked_log_prob.sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
        return per_anchor[has_positive].mean()

    def forward(self, z_list, labels):
        """z_list: list of num_boundaries tensors, each [B, C] (the
        boundary-specific pooled features model.py's BoundaryCornHead
        returns). labels: [B] long stage indices."""
        binary_targets = boundary_binary_targets(labels, len(z_list))  # [B, K]
        per_boundary = [
            self._single_boundary_loss(z_k, binary_targets[:, k])
            for k, z_k in enumerate(z_list)
        ]
        return torch.stack(per_boundary).mean()
