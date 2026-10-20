#!/usr/bin/env python3
"""Architecture-level checks for exp-5's residual fusion model, both
variants ("rgb_tissue" = original, "tissue_only" = new). Don't need a real
RGB baseline checkpoint (checked separately, against real data, at the
top of train.py once a checkpoint exists).

Checks per variant:
  1. The residual head's final layer (delta_proj) is exactly zero at
     construction (weight AND bias).
  2. delta_z == 0 for several different random inputs, not just one.
  3. z_final == z_rgb exactly at construction, for several different
     RGB-branch states (random re-inits), not just one.
  4. After a manual nonzero write to delta_proj's weights (simulating
     "after some training"), delta_z is no longer zero.
  5. Gradients reach the residual head's earlier (non-zero) layers despite
     delta_proj starting at zero, across 3 successive backward steps.

tissue_only-specific check:
  6. delta_z depends ONLY on tissue, never on RGB: holding tissue fixed
     and changing RGB changes z_rgb but leaves delta_z byte-identical --
     the structural guarantee this variant exists for (not just "the
     branch tends not to use RGB", provably cannot).
"""
import sys
from pathlib import Path

import torch
from coral_pytorch.losses import corn_loss

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))
sys.path.insert(0, str(THIS_DIR))

from model import ResidualFusionHead, TissueOnlyResidualHead, build_model  # noqa: E402

NUM_CLASSES = 4
H = W = 224


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def run_checks_for(variant):
    print(f"\n--- variant={variant} ---")
    all_ok = True

    if variant == "rgb_tissue":
        head = ResidualFusionHead(rgb_dim=512, tissue_dim=128, num_classes=NUM_CLASSES)
        head_inputs = lambda: (torch.randn(4, 512), torch.randn(4, 128))  # noqa: E731
    else:
        head = TissueOnlyResidualHead(tissue_dim=128, num_classes=NUM_CLASSES)
        head_inputs = lambda: (torch.randn(4, 128),)  # noqa: E731

    all_ok &= check("delta_proj weight is exactly zero at construction",
                     torch.equal(head.delta_proj.weight, torch.zeros_like(head.delta_proj.weight)))
    all_ok &= check("delta_proj bias is exactly zero at construction",
                     torch.equal(head.delta_proj.bias, torch.zeros_like(head.delta_proj.bias)))

    torch.manual_seed(0)
    for trial in range(3):
        delta_z = head(*head_inputs())
        all_ok &= check(f"delta_z == 0 for random input set {trial}",
                         torch.equal(delta_z, torch.zeros_like(delta_z)))

    for trial in range(2):
        torch.manual_seed(trial + 10)
        model = build_model("mit_b1", "imagenet", num_classes=NUM_CLASSES, variant=variant)
        model.eval()
        rgb = torch.randn(2, 3, H, W)
        tissue = torch.rand(2, 5, H, W)
        with torch.no_grad():
            z_final, z_rgb, delta_z = model(rgb, tissue)
        all_ok &= check(f"z_final == z_rgb exactly at construction (trial {trial})",
                         torch.equal(z_final, z_rgb))
        all_ok &= check(f"delta_z == 0 at construction (trial {trial})",
                         torch.equal(delta_z, torch.zeros_like(delta_z)))

    with torch.no_grad():
        head.delta_proj.weight.fill_(0.1)
    delta_z_after = head(*head_inputs())
    all_ok &= check("after a manual nonzero write, delta_z is no longer zero (zero-init isn't a permanent freeze)",
                     delta_z_after.abs().sum().item() > 0)

    torch.manual_seed(0)
    model = build_model("mit_b1", "imagenet", num_classes=NUM_CLASSES, variant=variant)
    model.train()
    rgb = torch.randn(2, 3, H, W)
    tissue = torch.rand(2, 5, H, W)
    targets = torch.tensor([0, 3])
    optimizer = torch.optim.AdamW(model.residual_head.parameters(), lr=0.01)
    early_layer_grad_seen = False
    for step in range(3):
        optimizer.zero_grad()
        z_final, _, _ = model(rgb, tissue)
        loss = corn_loss(z_final, targets, num_classes=NUM_CLASSES)
        loss.backward()
        first_mlp_layer = model.residual_head.mlp[0]
        if first_mlp_layer.weight.grad is not None and first_mlp_layer.weight.grad.abs().sum().item() > 0:
            early_layer_grad_seen = True
        optimizer.step()
    all_ok &= check("gradients reach the residual head's earlier layers within 3 steps "
                     "(zero-init delta_proj doesn't permanently block upstream gradients)",
                     early_layer_grad_seen)

    if variant == "tissue_only":
        torch.manual_seed(0)
        model = build_model("mit_b1", "imagenet", num_classes=NUM_CLASSES, variant=variant)
        # Give the residual head nonzero weights first -- at init delta_z=0
        # regardless of input, which would make this check vacuous.
        with torch.no_grad():
            model.residual_head.delta_proj.weight.fill_(0.05)
        model.eval()
        tissue = torch.rand(2, 5, H, W)
        rgb_a = torch.randn(2, 3, H, W)
        rgb_b = torch.randn(2, 3, H, W) * 5 + 2  # deliberately very different RGB
        with torch.no_grad():
            _, z_rgb_a, delta_z_a = model(rgb_a, tissue)
            _, z_rgb_b, delta_z_b = model(rgb_b, tissue)
        all_ok &= check("changing RGB (same tissue) changes z_rgb",
                         not torch.equal(z_rgb_a, z_rgb_b))
        all_ok &= check("...but delta_z is BYTE-IDENTICAL regardless of RGB (structurally tissue-only)",
                         torch.equal(delta_z_a, delta_z_b))

    return all_ok


def main():
    all_ok = run_checks_for("rgb_tissue") and run_checks_for("tissue_only")
    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
