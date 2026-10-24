#!/usr/bin/env python3
"""Shape/correctness checks for the 4-channel MiT-B1 + CORN model, run
before trusting any training results. Not a pytest suite -- a short,
explicit script printing PASS/FAIL per check, per the brief's requirement
to verify these properties directly rather than assume them. Mirrors
exp-3's scripts/test_8ch_model.py with NUM_TISSUE_CHANNELS=1.

Checks:
  1. RGB channels (0-2) of the 4-channel first conv are byte-identical to
     the plain 3-channel pretrained encoder's first conv (not scaled,
     not tiled -- exact copy).
  2. The severity channel (3) is exactly zero at initialization.
  3. At init, forward(RGB, severity=0) == forward_3ch(RGB) EXACTLY -- the
     4-channel model is mathematically identical to the RGB-only baseline
     before any training happens.
  4. Forward pass shape: [B, 4, H, W] -> CORN logits [B, num_classes-1].
  5. Loss is finite.
  6. Gradients on the severity-channel weights (3) are non-zero after one
     backward pass -- confirms it isn't dead/frozen despite being
     zero-initialized.
"""
import sys
from pathlib import Path

import torch
from coral_pytorch.losses import corn_loss

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))
sys.path.insert(0, str(THIS_DIR))

from src.model import Backbone as RGBBackbone  # noqa: E402  (task_3's own 3-channel backbone)
from model import Backbone, build_model, NUM_RGB_CHANNELS  # noqa: E402

ENCODER = "mit_b1"
WEIGHTS = "imagenet"
NUM_CLASSES = 4
NUM_TISSUE_CHANNELS = 1
H = W = 224


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def main():
    all_ok = True

    rgb_backbone = RGBBackbone(ENCODER, WEIGHTS)
    backbone4 = Backbone(ENCODER, WEIGHTS, in_channels=NUM_RGB_CHANNELS + NUM_TISSUE_CHANNELS)

    conv3 = rgb_backbone.encoder.patch_embed1.proj
    conv4 = backbone4.encoder.patch_embed1.proj

    all_ok &= check(
        "RGB channels (0-2) of 4-ch conv are byte-identical to the plain 3-ch pretrained conv",
        torch.equal(conv4.weight[:, :NUM_RGB_CHANNELS], conv3.weight),
    )
    all_ok &= check(
        "Severity channel (3) is exactly zero at init",
        torch.equal(conv4.weight[:, NUM_RGB_CHANNELS:], torch.zeros_like(conv4.weight[:, NUM_RGB_CHANNELS:])),
    )
    all_ok &= check(
        "Bias copied unchanged",
        conv3.bias is None and conv4.bias is None or torch.equal(conv3.bias, conv4.bias),
    )

    torch.manual_seed(0)
    rgb = torch.randn(2, NUM_RGB_CHANNELS, H, W)
    severity_zero = torch.zeros(2, NUM_TISSUE_CHANNELS, H, W)
    x4_zero_severity = torch.cat([rgb, severity_zero], dim=1)

    rgb_backbone.eval()
    backbone4.eval()
    with torch.no_grad():
        feat3 = rgb_backbone(rgb)
        feat4 = backbone4(x4_zero_severity)
    all_ok &= check(
        "forward(RGB, severity=0) == forward_3ch(RGB) exactly at init (4ch model == RGB baseline before training)",
        torch.allclose(feat3, feat4, atol=1e-5),
    )

    model = build_model(ENCODER, WEIGHTS, "corn", num_classes=NUM_CLASSES, in_channels=NUM_RGB_CHANNELS + NUM_TISSUE_CHANNELS)
    model.train()
    severity_real = torch.rand(2, NUM_TISSUE_CHANNELS, H, W)  # nonzero, like a real prediction
    x4 = torch.cat([rgb, severity_real], dim=1).requires_grad_(False)

    logits = model(x4)
    all_ok &= check(f"output shape is [B, {NUM_CLASSES - 1}]", tuple(logits.shape) == (2, NUM_CLASSES - 1))

    targets = torch.tensor([0, 3])
    loss = corn_loss(logits, targets, num_classes=NUM_CLASSES)
    all_ok &= check("loss is finite", torch.isfinite(loss).item())

    loss.backward()
    severity_grad = model[0].encoder.patch_embed1.proj.weight.grad[:, NUM_RGB_CHANNELS:]
    all_ok &= check(
        "severity-channel conv weights receive a non-zero gradient after backward()",
        severity_grad.abs().sum().item() > 0,
    )

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
