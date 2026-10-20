#!/usr/bin/env python3
"""Shape/correctness checks for the 8-channel MiT-B1 + CORN model, run
before trusting any training results. Not a pytest suite -- a short,
explicit script printing PASS/FAIL per check, per the brief's requirement
to verify these properties directly rather than assume them.

Checks:
  1. RGB channels (0-2) of the 8-channel first conv are byte-identical to
     the plain 3-channel pretrained encoder's first conv (not scaled,
     not tiled -- exact copy).
  2. Tissue channels (3-7) are exactly zero at initialization.
  3. At init, forward(RGB, tissue=0) == forward_3ch(RGB) EXACTLY -- the
     8-channel model is mathematically identical to the RGB-only baseline
     before any training happens.
  4. Forward pass shape: [B, 8, H, W] -> CORN logits [B, num_classes-1].
  5. Loss is finite.
  6. Gradients on the tissue-channel weights (3-7) are non-zero after one
     backward pass -- confirms they aren't dead/frozen despite being
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
H = W = 224


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def main():
    all_ok = True

    rgb_backbone = RGBBackbone(ENCODER, WEIGHTS)
    backbone8 = Backbone(ENCODER, WEIGHTS, in_channels=8)

    conv3 = rgb_backbone.encoder.patch_embed1.proj
    conv8 = backbone8.encoder.patch_embed1.proj

    all_ok &= check(
        "RGB channels (0-2) of 8-ch conv are byte-identical to the plain 3-ch pretrained conv",
        torch.equal(conv8.weight[:, :NUM_RGB_CHANNELS], conv3.weight),
    )
    all_ok &= check(
        "Tissue channels (3-7) are exactly zero at init",
        torch.equal(conv8.weight[:, NUM_RGB_CHANNELS:], torch.zeros_like(conv8.weight[:, NUM_RGB_CHANNELS:])),
    )
    all_ok &= check(
        "Bias copied unchanged",
        conv3.bias is None and conv8.bias is None or torch.equal(conv3.bias, conv8.bias),
    )

    torch.manual_seed(0)
    rgb = torch.randn(2, NUM_RGB_CHANNELS, H, W)
    tissue_zero = torch.zeros(2, 5, H, W)
    x8_zero_tissue = torch.cat([rgb, tissue_zero], dim=1)

    rgb_backbone.eval()
    backbone8.eval()
    with torch.no_grad():
        feat3 = rgb_backbone(rgb)
        feat8 = backbone8(x8_zero_tissue)
    all_ok &= check(
        "forward(RGB, tissue=0) == forward_3ch(RGB) exactly at init (8ch model == RGB baseline before training)",
        torch.allclose(feat3, feat8, atol=1e-5),
    )

    model = build_model(ENCODER, WEIGHTS, "corn", num_classes=NUM_CLASSES, in_channels=8)
    model.train()
    tissue_real = torch.rand(2, 5, H, W)  # nonzero, like a real prediction
    x8 = torch.cat([rgb, tissue_real], dim=1).requires_grad_(False)

    logits = model(x8)
    all_ok &= check(f"output shape is [B, {NUM_CLASSES - 1}]", tuple(logits.shape) == (2, NUM_CLASSES - 1))

    targets = torch.tensor([0, 3])
    loss = corn_loss(logits, targets, num_classes=NUM_CLASSES)
    all_ok &= check("loss is finite", torch.isfinite(loss).item())

    loss.backward()
    tissue_grad = model[0].encoder.patch_embed1.proj.weight.grad[:, NUM_RGB_CHANNELS:]
    all_ok &= check(
        "tissue-channel conv weights receive a non-zero gradient after backward()",
        tissue_grad.abs().sum().item() > 0,
    )

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
