"""Late-fusion model for exp-4: RGB and tissue channels processed by
SEPARATE encoders, combined after each has its own pooled feature vector
-- unlike exp-3 (early fusion), which concatenates at the input and lets
one shared conv stack learn both together.

  RGB arm:     F_rgb     = GAP(MiT-B1_ImageNet(rgb))        in R^512
  Tissue arm:  F_tissue  = GAP(SmallCNN(tissue_5ch))         in R^128

Two fusion modes (--fusion concat | gated):
  concat: Z = MLP([F_rgb ; F_tissue])  (640 -> 256 -> 256), CornHead(256)
  gated:  g = sigmoid(MLP_g([F_rgb ; F_tissue]))  in [0,1]^512
          F_fused = F_rgb + g (*) (W @ F_tissue)  -- W: Linear(128, 512)
          CornHead(512) directly on F_fused, no extra MLP (matches the
          brief's own notation: y_hat = CORN(Z) with Z = F_fused here)

RGB arm reuses task_3/src/model.py's Backbone UNCHANGED (ImageNet mit_b1,
3-channel, nothing about it is modified for this experiment) -- not the
local 8-channel Backbone from exp3_tissue_channels/scripts/model.py, which
doesn't apply here since RGB and tissue are never concatenated.

NOT initialized from a staging-finetuned RGB checkpoint: no such
checkpoint exists locally (task_3's own MiT-B1+CORN baseline has no saved
.pt, see more_experiments/README.md section 5) -- the RGB arm starts from
plain ImageNet weights, same starting point as every other experiment in
this project. Documented, not silently assumed away.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent  # exp/exp4_late_fusion/scripts/
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import Backbone as RGBBackbone, CornHead  # noqa: E402

NUM_TISSUE_CHANNELS = 5
TISSUE_FEATURE_DIM = 128


class TissueCNN(nn.Module):
    """3-conv-layer encoder for the 5 tissue-probability channels, global-
    average-pooled to a fixed-size feature vector. Deliberately small --
    ~1,091 training images is nowhere near enough to justify a second
    transformer-scale encoder for a 5-channel input."""

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


class GatedFusionHead(nn.Module):
    """g = sigmoid(MLP_g([F_rgb; F_tissue])); F_fused = F_rgb + g * (W @ F_tissue).
    Lets the model learn, per-sample, how much to trust the tissue
    prediction instead of always using it at full strength -- useful since
    tissue-segmentation quality isn't uniform across images (see the
    brief: a noisy Task 2 prediction on a given wound shouldn't be forced
    onto the staging decision at the same weight as a clean one)."""

    def __init__(self, rgb_dim, tissue_dim, num_classes, gate_hidden_dim=128):
        super().__init__()
        self.project = nn.Linear(tissue_dim, rgb_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(rgb_dim + tissue_dim, gate_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, rgb_dim), nn.Sigmoid(),
        )
        self.corn = CornHead(rgb_dim, num_classes)

    def forward(self, f_rgb, f_tissue):
        g = self.gate_mlp(torch.cat([f_rgb, f_tissue], dim=1))
        projected_tissue = self.project(f_tissue)
        f_fused = f_rgb + g * projected_tissue
        return self.corn(f_fused)


FUSION_HEADS = {"concat": ConcatFusionHead, "gated": GatedFusionHead}


class LateFusionModel(nn.Module):
    def __init__(self, encoder, weights, fusion, num_classes=4):
        super().__init__()
        self.rgb_backbone = RGBBackbone(encoder, weights)
        self.tissue_cnn = TissueCNN()
        self.fusion = FUSION_HEADS[fusion](self.rgb_backbone.out_dim, self.tissue_cnn.out_dim, num_classes)

    def forward(self, rgb, tissue):
        f_rgb = self.rgb_backbone(rgb)
        f_tissue = self.tissue_cnn(tissue)
        return self.fusion(f_rgb, f_tissue)

    def set_rgb_backbone_trainable(self, trainable):
        for p in self.rgb_backbone.parameters():
            p.requires_grad = trainable


def build_model(encoder, weights, fusion, num_classes=4):
    return LateFusionModel(encoder, weights, fusion, num_classes=num_classes)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
