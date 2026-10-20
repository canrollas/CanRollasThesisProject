"""8-channel backbone for exp-3 (RGB + 5 tissue-probability channels).

Reuses task_3/src/model.py's SoftmaxHead/CornHead unchanged (they only
look at the backbone's pooled feature vector, not input channel count).
Only the backbone differs: smp.encoders.get_encoder supports arbitrary
in_channels directly -- for a pretrained encoder it inflates the first
conv layer's ImageNet weights to the requested channel count instead of
discarding the pretraining, so MiT-B1 stays ImageNet-initialized even at
8 input channels. Verified working with a forward pass before relying on
it (in_channels=8, mit_b1, weights=imagenet -> correct output shapes).
"""
import sys
from pathlib import Path

import segmentation_models_pytorch as smp
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent  # exp/exp3_tissue_channels/scripts/
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import HEAD_MAP  # noqa: E402  (SoftmaxHead, CornHead -- reused as-is)

NUM_INPUT_CHANNELS = 8  # 3 RGB + 5 tissue (granulation, slough, necrosis, tendon, bone)


class Backbone(nn.Module):
    def __init__(self, encoder_name, weights, in_channels=NUM_INPUT_CHANNELS):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=in_channels, depth=5, weights=weights)
        self.out_dim = self.encoder.out_channels[-1]

    def forward(self, x):
        last_feature_map = self.encoder(x)[-1]
        return last_feature_map.mean(dim=(2, 3))


def build_model(encoder, weights, head, num_classes=4, in_channels=NUM_INPUT_CHANNELS):
    backbone = Backbone(encoder, weights, in_channels=in_channels)
    head_module = HEAD_MAP[head](backbone.out_dim, num_classes)
    return nn.Sequential(backbone, head_module)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
