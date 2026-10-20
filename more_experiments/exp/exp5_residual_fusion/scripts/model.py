"""Residual late fusion for exp-5, two variants.

Branch A (RGB): task_3/src/model.py's own Backbone + CornHead, UNCHANGED
architecture, loaded from the trained RGB baseline checkpoint (not
ImageNet-only, unlike exp-4) -- see load_rgb_baseline() below. ALWAYS
frozen (Phase 1).

Branch B (tissue): 3-conv-layer CNN on the 5 tissue-probability channels,
global-average-pooled. GroupNorm, not BatchNorm (see TissueCNN docstring
for why -- diagnosed after variant "rgb_tissue" below).

Two fusion variants (model.VARIANT or build_model(..., variant=...)):

  "rgb_tissue" (original exp-5 design): delta_z = ResidualFusionHead(f_rgb,
  f_tissue) -- the residual head sees BOTH branches concatenated.
  diagnose_residual.py found this doesn't work: cosine(delta_z_real,
  delta_z_shuffled_tissue) ~= 1.0 across all 3 folds at the converged
  checkpoint -- the branch produces almost the same correction regardless
  of which tissue map it's given, i.e. it isn't using tissue CONTENT.
  Plausible cause: with f_rgb (512-dim, confident/large-magnitude) and
  f_tissue (128-dim) concatenated into one MLP, gradient signal can route
  through the f_rgb half instead of forcing the model to extract anything
  from tissue -- the easiest way to reduce loss early in training doesn't
  have to involve tissue content at all.

  "tissue_only" (new): delta_z = TissueOnlyResidualHead(f_tissue) --
  the residual head NEVER sees f_rgb, only the tissue branch's own pooled
  features, so delta_z structurally CANNOT be a function of RGB content.
  Whatever correction it learns is forced to come from tissue information
  alone (or lack thereof) -- directly testable again with
  diagnose_residual.py's tissue-shuffle ablation.

Both keep the zero-initialized final residual projection (delta_z = 0 at
construction, model reproduces the frozen RGB baseline exactly before
training -- verified in scripts/test_residual_model.py and again in
train.py against real data) and both are trained the same way: Phase 1,
RGB branch frozen throughout, real early stopping on val macro-F1.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent  # exp/exp5_residual_fusion/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import Backbone as RGBBackbone, CornHead  # noqa: E402

NUM_TISSUE_CHANNELS = 5
TISSUE_FEATURE_DIM = 128


class TissueCNN(nn.Module):
    """3-conv-layer encoder for the 5 tissue-probability channels, global-
    average-pooled to a fixed-size feature vector -- deliberately small,
    ~1,091 training images is nowhere near enough to justify a second
    transformer-scale encoder for a 5-channel input.

    GroupNorm (8 groups), not BatchNorm: batch_size=16 on this dataset
    size means BatchNorm's running statistics are estimated from very few
    samples per step, which can make an already-weak gradient signal (see
    diagnose_residual.py's finding: ||grad|| on this branch was ~1e-3 to
    5e-3 at the converged "rgb_tissue" checkpoint) noisier still.
    GroupNorm's statistics are per-sample, not per-batch, removing that
    specific source of instability -- one isolated change at a time, not
    bundled with the fusion-head redesign, so if results change it's
    attributable."""

    def __init__(self, in_channels=NUM_TISSUE_CHANNELS, out_dim=TISSUE_FEATURE_DIM, num_groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.GroupNorm(num_groups, 32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.GroupNorm(num_groups, 64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1), nn.GroupNorm(num_groups, out_dim), nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x):
        feat = self.net(x)
        return feat.mean(dim=(2, 3))


class ResidualFusionHead(nn.Module):
    """Original "rgb_tissue" variant: delta_z = Linear_zero(ReLU(Linear(
    ReLU(Linear([f_rgb; f_tissue]))))) -- sees both branches. Kept
    (unmodified architecture, just now running on the GroupNorm TissueCNN
    if re-instantiated) for comparison against "tissue_only" below --
    NOT retrained here, the already-saved checkpoint/results for this
    variant are untouched."""

    def __init__(self, rgb_dim, tissue_dim, num_classes, hidden_dim=128, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(rgb_dim + tissue_dim, hidden_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.delta_proj = nn.Linear(hidden_dim, num_classes - 1)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(self, f_rgb, f_tissue):
        h = self.mlp(torch.cat([f_rgb, f_tissue], dim=1))
        return self.delta_proj(h)


class TissueOnlyResidualHead(nn.Module):
    """New "tissue_only" variant: delta_z = Linear_zero(ReLU(Linear(
    ReLU(Linear(f_tissue))))) -- f_rgb is NEVER an input here, structurally
    (not just empirically) forcing delta_z to be a function of tissue
    features alone. dropout=0.2 between the hidden layers (the
    "rgb_tissue" variant had none) to reduce the overfit-then-get-rejected-
    by-early-stopping pattern diagnose_residual.py found (train loss kept
    falling while val metrics degraded after epoch 1-2 in every fold).
    Final layer still zero-initialized -- delta_z = 0 at construction,
    gradient flow into the earlier layers despite that is checked in
    scripts/test_residual_model.py, same as before."""

    def __init__(self, tissue_dim, num_classes, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tissue_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.delta_proj = nn.Linear(hidden_dim, num_classes - 1)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(self, f_tissue):
        h = self.mlp(f_tissue)
        return self.delta_proj(h)


VARIANTS = ("rgb_tissue", "tissue_only")


class ResidualLateFusionModel(nn.Module):
    def __init__(self, encoder, weights, num_classes=4, variant="tissue_only"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.rgb_backbone = RGBBackbone(encoder, weights)
        self.rgb_head = CornHead(self.rgb_backbone.out_dim, num_classes)
        self.tissue_cnn = TissueCNN()
        if variant == "rgb_tissue":
            self.residual_head = ResidualFusionHead(self.rgb_backbone.out_dim, self.tissue_cnn.out_dim, num_classes)
        else:
            self.residual_head = TissueOnlyResidualHead(self.tissue_cnn.out_dim, num_classes)

    def forward(self, rgb, tissue):
        f_rgb = self.rgb_backbone(rgb)
        z_rgb = self.rgb_head(f_rgb)
        f_tissue = self.tissue_cnn(tissue)
        if self.variant == "rgb_tissue":
            delta_z = self.residual_head(f_rgb, f_tissue)
        else:
            delta_z = self.residual_head(f_tissue)
        z_final = z_rgb + delta_z
        return z_final, z_rgb, delta_z

    def set_rgb_branch_trainable(self, trainable):
        for p in self.rgb_backbone.parameters():
            p.requires_grad = trainable
        for p in self.rgb_head.parameters():
            p.requires_grad = trainable


def load_rgb_baseline(model, checkpoint_path, device):
    """Loads a task_3 baseline checkpoint (torch.save'd from
    nn.Sequential(Backbone, CornHead).state_dict() -- keys prefixed "0."
    for the backbone, "1." for the head) into model.rgb_backbone /
    model.rgb_head. strict=True: fails loudly on any key mismatch instead
    of silently loading a partial/wrong checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    backbone_state = {k[len("0."):]: v for k, v in state_dict.items() if k.startswith("0.")}
    head_state = {k[len("1."):]: v for k, v in state_dict.items() if k.startswith("1.")}
    if not backbone_state or not head_state:
        raise RuntimeError(
            f"{checkpoint_path}: expected keys prefixed '0.' (backbone) and '1.' (head) from an "
            "nn.Sequential(Backbone, CornHead) checkpoint, found none of one or both -- "
            f"got prefixes: {sorted({k.split('.')[0] for k in state_dict})}"
        )
    model.rgb_backbone.load_state_dict(backbone_state, strict=True)
    model.rgb_head.load_state_dict(head_state, strict=True)


def build_model(encoder, weights, num_classes=4, variant="tissue_only"):
    return ResidualLateFusionModel(encoder, weights, num_classes=num_classes, variant=variant)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
