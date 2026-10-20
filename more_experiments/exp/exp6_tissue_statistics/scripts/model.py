"""Models for exp-6 (hand-crafted tissue statistics).

StatsOnlyModel: small MLP + CornHead on the 20-dim feature vector alone --
no RGB input anywhere. Isolates whether the tissue probability maps carry
ANY staging signal independent of appearance.

StatsResidualModel: frozen RGB MiT-B1+CORN baseline (task_3/src/model.py's
own Backbone+CornHead, loaded from the trained checkpoint, exactly like
exp-5) + a small residual MLP on the SAME 20-dim feature vector, final
layer zero-initialized -- z_final = z_rgb + delta_z, identical mechanism
to exp-5's TissueOnlyResidualHead, just fed hand-crafted statistics
instead of CNN features. forward(rgb, stats) matches
exp5/scripts/engine.py's expected signature exactly, so that engine is
reused unchanged for training this.
"""
import sys
from pathlib import Path

import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent  # exp/exp6_tissue_statistics/scripts/
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import Backbone as RGBBackbone, CornHead  # noqa: E402

STATS_DIM = 20  # 5 tissue classes x 4 stats (mean, std, p95, frac_above_0.5)


class StatsOnlyModel(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=64, dropout=0.2, in_dim=STATS_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.head = CornHead(hidden_dim, num_classes)

    def forward(self, stats):
        return self.head(self.mlp(stats))


class StatsResidualHead(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=64, dropout=0.2, in_dim=STATS_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.delta_proj = nn.Linear(hidden_dim, num_classes - 1)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(self, stats):
        return self.delta_proj(self.mlp(stats))


class StatsResidualModel(nn.Module):
    def __init__(self, encoder, weights, num_classes=4):
        super().__init__()
        self.rgb_backbone = RGBBackbone(encoder, weights)
        self.rgb_head = CornHead(self.rgb_backbone.out_dim, num_classes)
        self.residual_head = StatsResidualHead(num_classes)

    def forward(self, rgb, stats):
        f_rgb = self.rgb_backbone(rgb)
        z_rgb = self.rgb_head(f_rgb)
        delta_z = self.residual_head(stats)
        z_final = z_rgb + delta_z
        return z_final, z_rgb, delta_z

    def set_rgb_branch_trainable(self, trainable):
        for p in self.rgb_backbone.parameters():
            p.requires_grad = trainable
        for p in self.rgb_head.parameters():
            p.requires_grad = trainable


def load_rgb_baseline(model, checkpoint_path, device):
    """Same loader as exp-5's -- nn.Sequential(Backbone, CornHead)
    checkpoint, keys prefixed "0."/"1.". strict=True: fails loudly on
    mismatch."""
    import torch
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


def build_stats_only_model(num_classes=4):
    return StatsOnlyModel(num_classes=num_classes)


def build_residual_model(encoder, weights, num_classes=4):
    return StatsResidualModel(encoder, weights, num_classes=num_classes)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
