#!/usr/bin/env python3
"""Architecture-only sanity check for exp12 -- no training, no GPU needed.
Verifies: Task 2's MAnet_mit_b2 checkpoint's encoder.* keys load into
task_3's own MiT-B2 Backbone with a fully matched key set, the weights
actually change from random init (not a silent no-op), and the resulting
model's forward pass produces CORN's expected [B, num_classes-1] logit
shape. Run locally before Colab (mirrors task_3/exp10's test_model.py
PASS/FAIL convention).
"""
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(THIS_DIR))
from model import build_model  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    torch.manual_seed(0)
    all_ok = True

    ckpt_path = (
        REPO_ROOT / "task_2/results/task_2_outputs/checkpoints/wound_inner_fold1/MAnet_mit_b2_imagenet.pth"
    )
    if not ckpt_path.is_file():
        print(
            f"SKIP: {ckpt_path} not found locally (Task 2's checkpoints are gitignored -- "
            f"this test only runs where Task 2's outputs are already present on disk)."
        )
        return

    model, verify = build_model(ckpt_path, torch.device("cpu"), num_classes=4)
    all_ok &= check(
        "all encoder.* keys present and shape-matched (332/332)",
        verify["checkpoint_encoder_keys"] == verify["backbone_encoder_keys"] == 332,
    )
    all_ok &= check("encoder weights changed from random init after load", verify["weights_changed_from_init"])

    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    all_ok &= check("CORN logits shape == [4, 3] (batch=4, num_classes-1=3)", tuple(logits.shape) == (4, 3))

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
