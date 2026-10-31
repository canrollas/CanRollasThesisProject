"""Work Package 1: single-tissue-channel late-fusion model, exp-4's
"concat" architecture unchanged except the tissue arm's input channel
count (5 -> 1). RGB arm, TissueCNN, and ConcatFusionHead are exp-4's own
scripts/model.py, copied here (not cross-imported, see exp-5's own
dataset.py docstring for why -- DataLoader multiprocessing pickling).
Only exp-4's "gated" variant is dropped (the brief leaves it as optional
follow-up, not part of this work package's required scope).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent  # experiment_extended/wp1_single_tissue_channel/exp4_concat/scripts/
REPO_ROOT = THIS_DIR.parents[4]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import Backbone as RGBBackbone, CornHead  # noqa: E402

TISSUE_FEATURE_DIM = 128


class TissueCNN(nn.Module):
    """exp-4's own 3-conv-layer tissue encoder, in_channels fixed to 1 --
    this work package only ever passes a single selected tissue channel."""

    def __init__(self, in_channels=1, out_dim=TISSUE_FEATURE_DIM):
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


class ConcatFusionHead(nn.Module):
    def __init__(self, rgb_dim, tissue_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(rgb_dim + tissue_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.corn = CornHead(hidden_dim, num_classes)

    def forward(self, f_rgb, f_tissue):
        z = self.mlp(torch.cat([f_rgb, f_tissue], dim=1))
        return self.corn(z)


class LateFusionModel(nn.Module):
    def __init__(self, encoder, weights, num_classes=4):
        super().__init__()
        self.rgb_backbone = RGBBackbone(encoder, weights)
        self.tissue_cnn = TissueCNN()
        self.fusion = ConcatFusionHead(self.rgb_backbone.out_dim, self.tissue_cnn.out_dim, num_classes)

    def forward(self, rgb, tissue):
        f_rgb = self.rgb_backbone(rgb)
        f_tissue = self.tissue_cnn(tissue)
        return self.fusion(f_rgb, f_tissue)

    def set_rgb_backbone_trainable(self, trainable):
        for p in self.rgb_backbone.parameters():
            p.requires_grad = trainable


def build_model(encoder, weights, num_classes=4):
    return LateFusionModel(encoder, weights, num_classes=num_classes)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
