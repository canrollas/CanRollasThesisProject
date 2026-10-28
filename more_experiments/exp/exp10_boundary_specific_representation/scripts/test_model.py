#!/usr/bin/env python3
"""Architecture-only tests for exp-10's BoundaryCornModel -- no real
checkpoint/training needed (uses a throwaway randomly-initialized
"0."/"1." state dict, exactly like a real baseline checkpoint's own key
layout, just to exercise load_rgb_baseline without touching real
checkpoints or results). Mirrors exp7/scripts/test_model.py's structure
and PASS/FAIL print convention.
"""
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from losses import BoundarySupConLoss, boundary_binary_targets  # noqa: E402
from model import (  # noqa: E402
    FEATURE_DIM, BoundaryCornModel, build_model, count_params_m, freeze_backbone, load_rgb_baseline,
)

sys.path.insert(0, str(THIS_DIR.parents[3] / "task_3"))
from src.model import build_model as build_baseline_model  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    torch.manual_seed(0)
    all_ok = True

    # 1. feature-map shape matches the spec (verified empirically, not hardcoded)
    model = build_model(mode="heads")
    model.eval()  # MiT-B1 has internal dropout layers -- without eval(), two
                  # separate forward passes on the same input would differ
    rgb = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        feat = model.backbone(rgb)
    all_ok &= check("final stage shape [B,512,7,7]", tuple(feat.shape) == (4, 512, 7, 7))
    all_ok &= check("FEATURE_DIM == 512", FEATURE_DIM == 512)

    # 2. "heads" mode: attention disabled -> every boundary's z_k equals
    #    plain GAP(F), the same vector the frozen baseline itself reads.
    with torch.no_grad():
        logits, baseline_logits, z_list = model(rgb)
    plain_gap = feat.mean(dim=(2, 3))
    all_ok &= check("'heads' mode: z_k == plain GAP(F) for every boundary (no attention)",
                     all(torch.allclose(z_k, plain_gap) for z_k in z_list))
    all_ok &= check("'heads' mode: 3 boundary logits produced", logits.shape == (4, 3))

    # 3. baseline-preserving sub-computation: build a throwaway baseline
    #    checkpoint, load it, assert the frozen baseline_logits sub-
    #    computation matches a reference CornHead EXACTLY.
    baseline = build_baseline_model("mit_b1", None, "corn", num_classes=4)
    tmp_ckpt = THIS_DIR.parent / "checkpoints" / "_test_throwaway_baseline.pt"
    tmp_ckpt.parent.mkdir(parents=True, exist_ok=True)  # checkpoints/ is empty -> not tracked by git,
                                                          # so a fresh clone (e.g. on Colab) doesn't have it yet
    torch.save(baseline.state_dict(), tmp_ckpt)

    for mode in ("heads", "attention", "contrastive"):
        model_m = build_model(mode=mode)
        load_rgb_baseline(model_m, tmp_ckpt, torch.device("cpu"))
        baseline.eval()
        model_m.eval()
        with torch.no_grad():
            reference_logits = baseline(rgb)
            reference_pooled = baseline[0](rgb)
            _logits, baseline_logits, _z = model_m(rgb)
            pooled = model_m.backbone(rgb).mean(dim=(2, 3))
        all_ok &= check(f"[{mode}] pooled GAP matches reference backbone exactly",
                         torch.equal(pooled, reference_pooled))
        all_ok &= check(f"[{mode}] frozen baseline_logits sub-computation matches reference CornHead exactly",
                         torch.equal(baseline_logits, reference_logits))
        all_ok &= check(f"[{mode}] baseline_head parameters are frozen (requires_grad=False)",
                         all(not p.requires_grad for p in model_m.head.baseline_head.parameters()))
    tmp_ckpt.unlink()

    # 4. attention mode: A_k is the constant 0.5 at construction (zero-init
    #    conv -> sigmoid(0)=0.5), so z_k != plain GAP(F) by a factor of 0.5,
    #    and attention_map() reports that constant directly.
    model_attn = build_model(mode="attention")
    with torch.no_grad():
        attn_map = model_attn.head.boundary_heads[0].attention_map(feat)
    all_ok &= check("'attention' mode: A_k constant 0.5 at construction",
                     torch.allclose(attn_map, torch.full_like(attn_map, 0.5)))

    # 5. gradient flow: a single backward pass must reach every attention
    #    conv's weight in "attention"/"contrastive" mode (the whole point
    #    of NOT zero-initializing the residual MLP's last layer -- see
    #    model.py's docstring).
    for mode in ("attention", "contrastive"):
        model_g = build_model(mode=mode)
        freeze_backbone(model_g)
        logits, _baseline_logits, z_list = model_g(rgb)
        loss = logits.sum() + sum(z.sum() for z in z_list)
        loss.backward()
        grads_nonzero = all(
            bh.gate.weight.grad is not None and bh.gate.weight.grad.abs().sum().item() > 0
            for bh in model_g.head.boundary_heads
        )
        all_ok &= check(f"[{mode}] gradient reaches every attention conv's weight after 1 backward pass",
                         grads_nonzero)
        backbone_has_no_grad = all(p.grad is None for p in model_g.backbone.parameters())
        all_ok &= check(f"[{mode}] frozen backbone receives no gradient", backbone_has_no_grad)

    # 6. freeze_backbone toggles requires_grad correctly, new heads stay trainable
    model_f = build_model(mode="attention")
    freeze_backbone(model_f)
    all_ok &= check("freeze_backbone: all backbone params frozen",
                     all(not p.requires_grad for p in model_f.backbone.parameters()))
    all_ok &= check("freeze_backbone: boundary_heads still trainable",
                     all(p.requires_grad for bh in model_f.head.boundary_heads for p in bh.parameters()))

    # 7. boundary_binary_targets: the 3 conditional splits CORN itself uses
    labels = torch.tensor([0, 1, 2, 3])  # stage_1..stage_4
    targets = boundary_binary_targets(labels, num_boundaries=3)
    expected = torch.tensor([
        [0, 0, 0],  # stage_1: past none of the 3 boundaries
        [1, 0, 0],  # stage_2: past boundary 0 only
        [1, 1, 0],  # stage_3: past boundaries 0,1
        [1, 1, 1],  # stage_4: past all 3
    ])
    all_ok &= check("boundary_binary_targets matches CORN's own conditional splits",
                     torch.equal(targets, expected))

    # 8. BoundarySupConLoss: degenerate single-class batch -> exactly 0, not NaN
    supcon = BoundarySupConLoss(temperature=0.1)
    z_single = [torch.randn(4, 16) for _ in range(3)]
    labels_single_class = torch.zeros(4, dtype=torch.long)  # all stage_1 -> every boundary's split is 1-sided
    loss_degenerate = supcon(z_single, labels_single_class)
    all_ok &= check("BoundarySupConLoss on an all-one-class batch is exactly 0 (not NaN)",
                     torch.equal(loss_degenerate, torch.zeros_like(loss_degenerate)))

    # 9. _single_boundary_loss: a single positive (group size 2 per side)
    #    perfectly separated across the binary split (orthogonal unit
    #    vectors, one direction per side) drives the loss to ~0 -- SupCon's
    #    per-anchor term is -(sim_pos - logsumexp(sim_pos, sim_negs,...)),
    #    which only -> 0 when there is exactly one positive AND it dominates
    #    the logsumexp; with more than one positive the asymptote is
    #    log(n_pos), not 0, so group size 2 (n_pos=1) is required to
    #    actually exercise the near-zero regime, not an arbitrary choice.
    binary_labels = torch.tensor([0, 0, 1, 1])
    e0 = torch.zeros(16); e0[0] = 1.0
    e1 = torch.zeros(16); e1[1] = 1.0
    z_separated = torch.stack([e0, e0, e1, e1])
    loss_separated = supcon._single_boundary_loss(z_separated, binary_labels)
    z_random_small = torch.randn(4, 16)
    loss_random_small = supcon._single_boundary_loss(z_random_small, binary_labels)
    all_ok &= check("_single_boundary_loss: perfectly-separated single-positive batch has near-zero loss",
                     loss_separated.item() < 1e-3)
    all_ok &= check("_single_boundary_loss: random-embedding batch loss is finite and > separated-batch loss",
                     bool(torch.isfinite(loss_random_small)) and loss_random_small.item() > loss_separated.item())

    print(f"\nparams: {count_params_m(build_model(mode='contrastive')):.3f}M total")
    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
