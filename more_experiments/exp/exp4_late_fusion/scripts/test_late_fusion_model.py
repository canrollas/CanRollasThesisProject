#!/usr/bin/env python3
"""Shape/correctness checks for exp-4's late-fusion model, both --fusion
modes, run before trusting any training result.

Checks per fusion mode:
  1. Forward pass shape: rgb [B,3,H,W] + tissue [B,5,H,W] -> CORN logits
     [B, num_classes-1].
  2. Loss is finite.
  3. Gradients reach the tissue CNN (it isn't a dead branch the fusion
     head could learn to ignore entirely at the gradient level).
  4. Gradients reach the RGB backbone too, by default (freeze_epochs=0
     means it's trainable from the start).
  5. set_rgb_backbone_trainable(False) actually stops gradients from
     reaching the RGB backbone (the freeze half of the staged
     freeze/fine-tune option actually works, not just flips a flag that's
     never checked).
"""
import sys
from pathlib import Path

import torch
from coral_pytorch.losses import corn_loss

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))
sys.path.insert(0, str(THIS_DIR))

from model import build_model  # noqa: E402

ENCODER = "mit_b1"
WEIGHTS = "imagenet"
NUM_CLASSES = 4
H = W = 224


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def run_checks_for(fusion):
    print(f"\n--- fusion={fusion} ---")
    all_ok = True
    torch.manual_seed(0)
    model = build_model(ENCODER, WEIGHTS, fusion, num_classes=NUM_CLASSES)
    model.train()

    rgb = torch.randn(2, 3, H, W)
    tissue = torch.rand(2, 5, H, W)
    targets = torch.tensor([0, 3])

    logits = model(rgb, tissue)
    all_ok &= check(f"output shape is [B, {NUM_CLASSES - 1}]", tuple(logits.shape) == (2, NUM_CLASSES - 1))

    loss = corn_loss(logits, targets, num_classes=NUM_CLASSES)
    all_ok &= check("loss is finite", torch.isfinite(loss).item())

    model.zero_grad()
    loss.backward()
    tissue_grad_sum = sum(p.grad.abs().sum().item() for p in model.tissue_cnn.parameters() if p.grad is not None)
    rgb_grad_sum = sum(p.grad.abs().sum().item() for p in model.rgb_backbone.parameters() if p.grad is not None)
    all_ok &= check("tissue CNN receives non-zero gradients", tissue_grad_sum > 0)
    all_ok &= check("RGB backbone receives non-zero gradients (freeze_epochs=0 default)", rgb_grad_sum > 0)

    model.set_rgb_backbone_trainable(False)
    model.zero_grad()
    logits2 = model(rgb, tissue)
    loss2 = corn_loss(logits2, targets, num_classes=NUM_CLASSES)
    loss2.backward()
    rgb_grad_after_freeze = [p.grad for p in model.rgb_backbone.parameters()]
    all_ok &= check("freezing the RGB backbone actually stops its gradients",
                     all(g is None for g in rgb_grad_after_freeze))

    return all_ok


def main():
    all_ok = run_checks_for("concat") and run_checks_for("gated")
    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
