"""Experiment 10 (Chapter 8): boundary-specific representation learning
for CORN.

Motivation: task_3/src/model.py's Backbone.forward pools the MiT-B1
encoder's final feature map into ONE 512-dim vector
(`last_feature_map.mean(dim=(2,3))`) BEFORE CornHead ever sees it, and
CornHead is a single `Linear(512,3)` -- so the three ordinal boundary
decisions ("is this past stage 1?", "past stage 2?", "past stage 3?")
all read the exact same pooled representation; only the three linear
weight ROWS differ. This experiment gives each boundary its own learned
spatial attention over the (un-pooled) feature map, so each boundary can
pool a different image region before its own decision head reads it --
without touching the dataset, labels, fold split, or the backbone's own
training.

Three ablation modes share one architecture (`BoundaryCornHead`), all
built on top of the FROZEN, already-trained RGB MiT-B1+CORN baseline --
same per-fold checkpoint exp5/6/7 load from
task_3/results/task_3_outputs/checkpoints/mit_b1__corn_fold{fold}.pt,
Phase A convention: only the new per-boundary heads train, the backbone
and the baseline's own CornHead never move. This isolates whatever the
new head contributes from any change to the backbone itself, and keeps
every mode directly comparable (same frozen inputs, same optimizer
budget, only the head differs).

  "heads"       Each boundary k gets its OWN small residual MLP on the
                SAME globally-pooled feature (attention gate A_k fixed
                at 1 everywhere -- every boundary and the baseline all
                read the identical pooled vector). Tests whether letting
                each boundary have a nonlinear, boundary-specific
                DECISION function helps, before touching spatial
                attention at all -- the user's own first move, since
                "just add 3 heads" is not obviously different from the
                baseline's existing 3 independent weight rows and must
                be checked before anything fancier.
  "attention"   Adds the learned per-boundary spatial gate
                A_k = sigmoid(Conv1x1_k(F)); z_k = GAP(F * A_k) replaces
                the fixed global pooling as each MLP's input. Tests
                whether letting each boundary SELECT different image
                regions helps.
  "contrastive" Same architecture as "attention", trained with an added
                per-boundary supervised contrastive term on z_k (see
                losses.py) that explicitly pulls same-side images
                together and pushes opposite-side images apart in that
                boundary's own representation space.

logit_k = baseline_logit_k + delta_logit_k. baseline_logit_k is row k of
the FROZEN, loaded CornHead applied to the unmodified GAP(F) -- exactly
task_3's own baseline computation, always present, never trained here.
delta_logit_k is boundary k's own residual head's output.

Unlike exp5/6/7's additive residual (whose final layer is
zero-initialized so the fused model reproduces the baseline EXACTLY at
construction), this residual head's final Linear layer uses PyTorch's
ordinary init, not zero: a zero-initialized last layer
only receives nonzero GRADIENT at step 1 (outer-gradient times its own
input); the layers and the attention conv BEFORE it still see a zero
upstream gradient at that same step (multiplied through the zero weight
matrix after them), delaying when the attention module starts learning
anything at all. The tradeoff, stated plainly: at construction, this
model's logits do NOT exactly equal the frozen baseline's logits (unlike
exp5/6/7). This is intentional, not a defect -- what IS verified directly
in test_model.py: the frozen baseline sub-computation itself exactly
reproduces a reference CornHead's output, the backbone never receives a
gradient (Phase A, same as exp5/6/7), and gradient reaches every
attention conv's weight after a single real backward pass, in every
mode.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent  # exp/exp10_boundary_specific_representation/scripts/
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import Backbone as RGBBackbone, CornHead  # noqa: E402

FEATURE_DIM = 512  # mit_b1 final-stage channels, verified empirically in test_model.py
MODES = ("heads", "attention", "contrastive")


class MultiStageBackbone(nn.Module):
    """Wraps task_3's own Backbone (so a baseline checkpoint's "0." keys
    load into it unchanged) but exposes the raw final feature map
    [B,512,7,7] instead of only the pre-pooled GAP vector
    Backbone.forward returns."""

    def __init__(self, encoder_name="mit_b1", weights=None):
        super().__init__()
        self.inner = RGBBackbone(encoder_name, weights)  # has .encoder, .out_dim, .kind
        assert self.inner.kind == "smp"
        assert self.inner.out_dim == FEATURE_DIM

    def forward(self, x):
        return self.inner.encoder(x)[-1]  # [B, 512, 7, 7]


class BoundaryResidualHead(nn.Module):
    """One boundary's attention gate + residual decision head.

    attention=False ("heads" mode): A_k is fixed at 1 everywhere, so
    `pooled_feature` reduces to plain GAP(F) -- the SAME vector every
    boundary (and the frozen baseline) reads. Only the MLP after pooling
    is boundary-specific in this mode.

    attention=True ("attention"/"contrastive" modes): A_k =
    sigmoid(Conv1x1(F)), with the conv zero-initialized so A_k is the
    constant 0.5 everywhere at construction -- a non-degenerate start:
    sigmoid's gradient at input 0 is 0.25 (not 0), so the gate begins
    learning from step 1, it simply starts from "weight every location
    equally" rather than favoring any region a priori.
    """

    def __init__(self, in_channels=FEATURE_DIM, hidden_dim=128, attention=False):
        super().__init__()
        self.attention = attention
        if attention:
            self.gate = nn.Conv2d(in_channels, 1, kernel_size=1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def attention_map(self, feature_map):
        """A_k, [B,1,H,W]. Only meaningful when attention=True; returns a
        constant 0.5 map of the right shape otherwise (so callers -- e.g.
        diagnose_attention.py -- don't need an attention-mode branch)."""
        if self.attention:
            return torch.sigmoid(self.gate(feature_map))
        return feature_map.new_full((feature_map.shape[0], 1, *feature_map.shape[2:]), 0.5)

    def pooled_feature(self, feature_map):
        """z_k -- the boundary-specific pooled vector. Exposed separately
        from forward() so the contrastive loss (and diagnostics) can read
        it without a second forward pass through the MLP."""
        if self.attention:
            gate = torch.sigmoid(self.gate(feature_map))  # [B,1,H,W]
            weighted = feature_map * gate
        else:
            weighted = feature_map
        return weighted.mean(dim=(2, 3))  # [B, C]

    def forward(self, feature_map):
        z_k = self.pooled_feature(feature_map)
        delta_logit = self.mlp(z_k)  # [B, 1]
        return delta_logit, z_k


class BoundaryCornHead(nn.Module):
    """Three BoundaryResidualHeads (one per CORN boundary) added on top
    of the frozen baseline CornHead's own logits."""

    def __init__(self, in_dim=FEATURE_DIM, num_classes=4, hidden_dim=128, mode="heads"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        self.mode = mode
        self.num_boundaries = num_classes - 1
        attention = mode in ("attention", "contrastive")

        self.baseline_head = CornHead(in_dim, num_classes)
        for p in self.baseline_head.parameters():
            p.requires_grad = False

        self.boundary_heads = nn.ModuleList(
            BoundaryResidualHead(in_dim, hidden_dim, attention=attention)
            for _ in range(self.num_boundaries)
        )

    def forward(self, feature_map):
        pooled = feature_map.mean(dim=(2, 3))  # same op as the baseline Backbone's own GAP
        baseline_logits = self.baseline_head(pooled)  # [B, num_boundaries], frozen

        deltas, z_list = [], []
        for boundary_head in self.boundary_heads:
            delta_k, z_k = boundary_head(feature_map)
            deltas.append(delta_k)
            z_list.append(z_k)
        delta_logits = torch.cat(deltas, dim=1)  # [B, num_boundaries]

        logits = baseline_logits + delta_logits
        return logits, baseline_logits, z_list


class BoundaryCornModel(nn.Module):
    def __init__(self, encoder="mit_b1", weights=None, num_classes=4, hidden_dim=128, mode="heads"):
        super().__init__()
        self.backbone = MultiStageBackbone(encoder, weights)
        self.head = BoundaryCornHead(FEATURE_DIM, num_classes, hidden_dim, mode)

    def forward(self, x):
        feature_map = self.backbone(x)
        return self.head(feature_map)  # logits, baseline_logits, z_list


def load_rgb_baseline(model, checkpoint_path, device):
    """Loads an nn.Sequential(Backbone, CornHead) checkpoint (task_3's own
    save format): "0." keys -> model.backbone.inner, "1." keys ->
    model.head.baseline_head (frozen -- see BoundaryCornHead.__init__)."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    backbone_state = {k[len("0."):]: v for k, v in state_dict.items() if k.startswith("0.")}
    head_state = {k[len("1."):]: v for k, v in state_dict.items() if k.startswith("1.")}
    if not backbone_state or not head_state:
        raise RuntimeError(
            f"{checkpoint_path}: expected keys prefixed '0.' (backbone) and '1.' (head), "
            f"got prefixes: {sorted({k.split('.')[0] for k in state_dict})}"
        )
    model.backbone.inner.load_state_dict(backbone_state, strict=True)
    model.head.baseline_head.load_state_dict(head_state, strict=True)


def build_model(encoder="mit_b1", weights=None, num_classes=4, hidden_dim=128, mode="heads"):
    return BoundaryCornModel(encoder, weights, num_classes, hidden_dim, mode)


def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_params_m(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
