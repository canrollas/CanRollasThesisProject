#!/usr/bin/env python3
"""Exp13 model: DINOv2 ViT-L/14 (timm `vit_large_patch14_dinov2.lvd142m`,
303M params, self-supervised on LVD-142M -- no wound/medical data in its
pretraining) as the encoder, task_3's own CornHead on top. Not an
SMP/task_3-registered encoder (ViT, not CNN/MiT), so this exp builds the
model standalone via timm directly rather than extending
task_3/src/model.py's Backbone.

Two modes, chosen at construction:
  - freeze_backbone=True  ("linear probe"): encoder forced into eval() and
    run under torch.no_grad() every forward pass -- only CornHead trains.
    Cheap, safe default, feasible on CPU/MPS.
  - freeze_backbone=False ("fine-tune"): encoder trains too. 303M params
    on ~700 training images is a real overfitting/compute risk -- see
    README's "Fine-tune güvenliği" section. Use a much smaller lr than
    the frozen mode's (configs/base.yaml keeps them separate) and prefer
    Colab/GPU.

Images resized to 224x224 (same as task_3's own recipe, for direct
comparability) -- 224 / patch14 = 16, an exact 16x16 patch grid, verified
to forward cleanly (see README's "Mimari doğrulama").
"""
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import CornHead  # noqa: E402 -- reused unchanged, not modified

ENCODER_NAME = "vit_large_patch14_dinov2.lvd142m"


class DinoCornModel(nn.Module):
    def __init__(self, num_classes=4, freeze_backbone=True, img_size=224, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(ENCODER_NAME, pretrained=pretrained, num_classes=0, img_size=img_size)
        self.out_dim = self.encoder.num_features
        self.head = CornHead(self.out_dim, num_classes)
        self.freeze_backbone_flag = None
        self.set_freeze(freeze_backbone)

    def set_freeze(self, freeze):
        self.freeze_backbone_flag = freeze
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def forward(self, x):
        if self.freeze_backbone_flag:
            self.encoder.eval()
            with torch.no_grad():
                feat = self.encoder(x)
        else:
            feat = self.encoder(x)
        return self.head(feat)


def build_model(mode, num_classes=4, img_size=224, pretrained=True):
    if mode not in ("frozen", "finetune"):
        raise ValueError(f"unknown mode {mode!r}, expected 'frozen' or 'finetune'")
    return DinoCornModel(num_classes=num_classes, freeze_backbone=(mode == "frozen"),
                          img_size=img_size, pretrained=pretrained)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_params_m(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
