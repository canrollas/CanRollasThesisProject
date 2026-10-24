"""4-channel backbone for exp-3.1 (RGB + 1 severity-banded tissue channel).

Identical mechanism to exp-3's scripts/model.py (same file, just
NUM_TISSUE_CHANNELS=1 instead of 5) -- see that file's docstring for the
full first-conv-expansion rationale. Reuses task_3/src/model.py's
SoftmaxHead/CornHead unchanged (they only look at the backbone's pooled
feature vector, not input channel count).

First-layer expansion is EXPLICIT, not SMP's automatic in_channels
mechanism (that path was checked and rejected in exp-3 for confounding
the RGB-vs-RGB+tissue comparison -- see exp-3's model.py docstring).
Instead: copy the pretrained RGB weights into channels 0-2 UNCHANGED, and
zero-initialize channel 3 (the severity channel). At initialization this
makes the 4-channel model's first layer mathematically IDENTICAL to the
plain RGB baseline (the zero-weighted channel contributes exactly 0 to the
conv output) -- any divergence that emerges during training is then
attributable to the model learning to use the severity channel, not to a
different starting point for the RGB pathway. Zero-initialized weights
still receive gradients during backprop, verified directly in
scripts/test_4ch_model.py, not assumed.
"""
import sys
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent  # exp/exp3_1_tissue_single_channel/scripts/
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import HEAD_MAP  # noqa: E402  (SoftmaxHead, CornHead -- reused as-is)

NUM_RGB_CHANNELS = 3
NUM_TISSUE_CHANNELS = 1
NUM_INPUT_CHANNELS = NUM_RGB_CHANNELS + NUM_TISSUE_CHANNELS  # 4


def expand_first_conv(old_conv, extra_channels):
    """Returns a new Conv2d with the same kernel_size/stride/padding/
    dilation/groups/bias as old_conv, in_channels extended by
    extra_channels: the original (RGB) weights copied unchanged into the
    first NUM_RGB_CHANNELS, the new (tissue) channels zero-initialized."""
    new_in_channels = old_conv.in_channels + extra_channels
    new_conv = nn.Conv2d(
        new_in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, dilation=old_conv.dilation,
        groups=old_conv.groups, bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :old_conv.in_channels, :, :] = old_conv.weight
        new_conv.weight[:, old_conv.in_channels:, :, :] = 0.0
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    return new_conv


class Backbone(nn.Module):
    def __init__(self, encoder_name, weights, in_channels=NUM_INPUT_CHANNELS):
        super().__init__()
        # Always build the ORIGINAL 3-channel pretrained encoder first --
        # this is the actual ImageNet checkpoint load, verified identical
        # to the RGB-only baseline's own encoder construction.
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=NUM_RGB_CHANNELS, depth=5, weights=weights)

        extra_channels = in_channels - NUM_RGB_CHANNELS
        if extra_channels > 0:
            if not hasattr(self.encoder, "patch_embed1") or not hasattr(self.encoder.patch_embed1, "proj"):
                raise AttributeError(
                    f"{encoder_name} has no .patch_embed1.proj -- this expansion path is MiT-specific "
                    "(verified for mit_b1/mit_b2/mit_b3); check the attribute name before using "
                    "a different encoder family here."
                )
            self.encoder.patch_embed1.proj = expand_first_conv(self.encoder.patch_embed1.proj, extra_channels)

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
