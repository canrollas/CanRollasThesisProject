"""Baseline-preserving residual late fusion for exp-5.

Branch A (RGB): task_3/src/model.py's own Backbone + CornHead, UNCHANGED
architecture, loaded from the trained RGB baseline checkpoint (not
ImageNet-only, unlike exp-4) -- see load_rgb_baseline() below.

Branch B (tissue): reuses exp-4's TissueCNN directly (not reimplemented --
"avoid unnecessary refactoring").

Residual fusion: z_final = z_rgb + delta_z, where delta_z comes from a
small MLP whose FINAL linear layer is explicitly zero-initialized (weight
and bias both zero). This is the critical property the brief requires:
at construction, delta_z = 0 for ANY input regardless of what the earlier
MLP layers compute, so z_final == z_rgb exactly -- the fusion model
reproduces the frozen RGB baseline's predictions before a single training
step happens. scripts/test_residual_model.py checks this directly (not
just at the chosen input, across several random inputs) before any
training is trusted, and train.py re-checks it once more against the
actual loaded baseline checkpoint on real data before training starts.
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
    """Intentionally a direct copy of exp-4's TissueCNN (not a
    cross-module import -- see dataset.py's docstring for why: a
    dynamically-loaded module can break multiprocessing DataLoader
    workers' pickling; keeping both model.py's architecture definitions
    independently importable avoids that class of surprise entirely, not
    just in the one place it was first hit). 3-conv-layer encoder for the
    5 tissue-probability channels, global-average-pooled to a fixed-size
    feature vector -- deliberately small, ~1,091 training images is
    nowhere near enough to justify a second transformer-scale encoder for
    a 5-channel input."""

    def __init__(self, in_channels=NUM_TISSUE_CHANNELS, out_dim=TISSUE_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x):
        feat = self.net(x)
        return feat.mean(dim=(2, 3))


class ResidualFusionHead(nn.Module):
    """delta_z = Linear_zero(ReLU(Linear(ReLU(Linear([f_rgb; f_tissue])))))
    -- the LAST linear layer is zero-initialized, so delta_z = 0 at
    construction regardless of the earlier layers' (non-zero,
    randomly-initialized) weights."""

    def __init__(self, rgb_dim, tissue_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(rgb_dim + tissue_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.delta_proj = nn.Linear(hidden_dim, num_classes - 1)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(self, f_rgb, f_tissue):
        h = self.mlp(torch.cat([f_rgb, f_tissue], dim=1))
        return self.delta_proj(h)


class ResidualLateFusionModel(nn.Module):
    def __init__(self, encoder, weights, num_classes=4):
        super().__init__()
        self.rgb_backbone = RGBBackbone(encoder, weights)
        self.rgb_head = CornHead(self.rgb_backbone.out_dim, num_classes)
        self.tissue_cnn = TissueCNN()
        self.residual_head = ResidualFusionHead(self.rgb_backbone.out_dim, self.tissue_cnn.out_dim, num_classes)

    def forward(self, rgb, tissue):
        f_rgb = self.rgb_backbone(rgb)
        z_rgb = self.rgb_head(f_rgb)
        f_tissue = self.tissue_cnn(tissue)
        delta_z = self.residual_head(f_rgb, f_tissue)
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


def build_model(encoder, weights, num_classes=4):
    return ResidualLateFusionModel(encoder, weights, num_classes=num_classes)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
