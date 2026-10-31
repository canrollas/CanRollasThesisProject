"""Work Package 1: single-tissue-channel residual fusion model, exp-5's
"tissue_only" variant unchanged except the tissue arm's input channel
count (5 -> 1). RGB branch (frozen, loaded from the trained baseline
checkpoint), GroupNorm TissueCNN, TissueOnlyResidualHead, and
load_rgb_baseline() are exp-5's own scripts/model.py, copied here (not
cross-imported -- DataLoader multiprocessing pickling, same reasoning as
exp-5's own dataset.py docstring). The "rgb_tissue" variant is dropped:
exp-5's own diagnose_residual.py already found it doesn't use tissue
content (cosine(real,shuffled) ~= 1.0), so there is no reason to repeat it
per-channel here.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent  # experiment_extended/wp1_single_tissue_channel/exp5_tissue_only/scripts/
REPO_ROOT = THIS_DIR.parents[4]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import Backbone as RGBBackbone, CornHead  # noqa: E402

TISSUE_FEATURE_DIM = 128


class TissueCNN(nn.Module):
    """exp-5's own GroupNorm tissue encoder, in_channels fixed to 1."""

    def __init__(self, in_channels=1, out_dim=TISSUE_FEATURE_DIM, num_groups=8):
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


class TissueOnlyResidualHead(nn.Module):
    """delta_z = Linear_zero(ReLU(Linear(ReLU(Linear(f_tissue))))) -- f_rgb
    is never an input, same as exp-5's own tissue_only head. Final layer
    zero-initialized: delta_z = 0 at construction."""

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


class ResidualLateFusionModel(nn.Module):
    def __init__(self, encoder, weights, num_classes=4):
        super().__init__()
        self.rgb_backbone = RGBBackbone(encoder, weights)
        self.rgb_head = CornHead(self.rgb_backbone.out_dim, num_classes)
        self.tissue_cnn = TissueCNN()
        self.residual_head = TissueOnlyResidualHead(self.tissue_cnn.out_dim, num_classes)

    def forward(self, rgb, tissue):
        f_rgb = self.rgb_backbone(rgb)
        z_rgb = self.rgb_head(f_rgb)
        f_tissue = self.tissue_cnn(tissue)
        delta_z = self.residual_head(f_tissue)
        z_final = z_rgb + delta_z
        return z_final, z_rgb, delta_z

    def set_rgb_branch_trainable(self, trainable):
        for p in self.rgb_backbone.parameters():
            p.requires_grad = trainable
        for p in self.rgb_head.parameters():
            p.requires_grad = trainable


def load_rgb_baseline(model, checkpoint_path, device):
    """Same "0."/"1." key-split convention as exp-5's own model.py (the
    checkpoint is an nn.Sequential(Backbone, CornHead).state_dict())."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    backbone_state = {k[len("0."):]: v for k, v in state_dict.items() if k.startswith("0.")}
    head_state = {k[len("1."):]: v for k, v in state_dict.items() if k.startswith("1.")}
    if not backbone_state or not head_state:
        raise RuntimeError(
            f"{checkpoint_path}: expected keys prefixed '0.' (backbone) and '1.' (head), "
            f"got prefixes: {sorted({k.split('.')[0] for k in state_dict})}"
        )
    model.rgb_backbone.load_state_dict(backbone_state, strict=True)
    model.rgb_head.load_state_dict(head_state, strict=True)


def build_model(encoder, weights, num_classes=4):
    return ResidualLateFusionModel(encoder, weights, num_classes=num_classes)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
