"""Three-class RGB baseline and Exp 4 concat fusion, both initialized afresh."""
import sys
from pathlib import Path

import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "task_3"))
from src.model import Backbone as RGBBackbone, CornHead  # noqa: E402
from more_experiments.exp.exp4_late_fusion.scripts.model import (  # noqa: E402
    ConcatFusionHead, TissueCNN,
)

VARIANTS = ("rgb", "rgb_tissue")


class Stage234Model(nn.Module):
    def __init__(self, variant, weights="imagenet", encoder="mit_b1"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.rgb_backbone = RGBBackbone(encoder, weights)
        if variant == "rgb":
            self.head = CornHead(self.rgb_backbone.out_dim, num_classes=3)
        else:
            self.tissue_cnn = TissueCNN()
            self.fusion = ConcatFusionHead(self.rgb_backbone.out_dim, self.tissue_cnn.out_dim, num_classes=3)

    def forward(self, rgb, tissue):
        features = self.rgb_backbone(rgb)
        if self.variant == "rgb":
            return self.head(features)
        return self.fusion(features, self.tissue_cnn(tissue))


def build_variant(variant, weights="imagenet", encoder="mit_b1"):
    return Stage234Model(variant, weights=weights, encoder=encoder)
