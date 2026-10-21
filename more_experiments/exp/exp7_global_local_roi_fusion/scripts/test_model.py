#!/usr/bin/env python3
"""Architecture-only tests for exp-7's GlobalLocalFusionModel -- no
checkpoint/training needed (uses a throwaway randomly-initialized "0."/"1."
state dict, exactly like the baseline checkpoint's own key layout, just to
exercise load_rgb_baseline/init_from_baseline without touching real
checkpoints or results)."""
import sys
from pathlib import Path

import torch
from torchvision.ops import roi_align

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from model import (  # noqa: E402
    GLOBAL_DIM, GlobalLocalFusionModel, LOCAL_IN_CHANNELS, build_model,
    count_params_m, freeze_backbone, load_rgb_baseline, unfreeze_last_stage,
)

sys.path.insert(0, str(THIS_DIR.parents[3] / "task_3"))
from src.model import build_model as build_baseline_model  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    torch.manual_seed(0)
    all_ok = True

    # 1. feature-map shapes match the spec (verified empirically, not hardcoded)
    model = build_model()
    rgb = torch.randn(2, 3, 224, 224)
    feat3, featF = model.backbone(rgb)
    all_ok &= check("stage3 shape [B,320,14,14]", tuple(feat3.shape) == (2, 320, 14, 14))
    all_ok &= check("final stage shape [B,512,7,7]", tuple(featF.shape) == (2, 512, 7, 7))
    all_ok &= check("LOCAL_IN_CHANNELS == 320", LOCAL_IN_CHANNELS == 320)
    all_ok &= check("GLOBAL_DIM == 512", GLOBAL_DIM == 512)

    # 2. baseline-preserving init: build a throwaway baseline checkpoint,
    #    load it, assert fusion logits == baseline logits EXACTLY at init.
    baseline = build_baseline_model("mit_b1", None, "corn", num_classes=4)
    tmp_ckpt = THIS_DIR.parent / "checkpoints" / "_test_throwaway_baseline.pt"
    torch.save(baseline.state_dict(), tmp_ckpt)

    model2 = build_model()
    load_rgb_baseline(model2, tmp_ckpt, torch.device("cpu"))
    tmp_ckpt.unlink()
    baseline.eval()
    model2.eval()

    boxes = [torch.tensor([[20.0, 20.0, 150.0, 150.0]]) for _ in range(2)]
    with torch.no_grad():
        baseline_logits = baseline(rgb)
        fusion_logits, f_global = model2(rgb, boxes)

    all_ok &= check("f_global matches baseline's own GAP output (torch.equal)",
                     torch.equal(f_global, baseline.__getitem__(0)(rgb)))
    all_ok &= check("fusion_logits == baseline_logits EXACTLY at init (torch.equal)",
                     torch.equal(fusion_logits, baseline_logits))

    # 3. fusion_logits are box-INVARIANT at init -- this is the CORRECT,
    #    expected consequence of zero-initializing the head's local columns
    #    (0 * f_local == 0 regardless of the box), not a bug. The real
    #    content-sensitivity check belongs on f_local itself, before the
    #    zero-multiplication (test 3b).
    boxes_other = [torch.tensor([[0.0, 0.0, 30.0, 30.0]]) for _ in range(2)]
    with torch.no_grad():
        fusion_logits_other, _ = model2(rgb, boxes_other)
    all_ok &= check("fusion_logits are box-invariant at init (expected: 0-weighted local columns)",
                     torch.equal(fusion_logits, fusion_logits_other))

    # 3b. f_local itself (pre-zero-multiplication) IS box-sensitive -- proves
    #     roi_align is reading real, box-dependent spatial content, so once
    #     the head's local columns start moving away from zero during
    #     training, there is an actual signal for them to pick up.
    with torch.no_grad():
        feat3_, _ = model2.backbone(rgb)
        scale = feat3_.shape[-1] / rgb.shape[-1]
        roi_a = roi_align(feat3_, boxes, output_size=(3, 3), spatial_scale=scale, aligned=True)
        roi_b = roi_align(feat3_, boxes_other, output_size=(3, 3), spatial_scale=scale, aligned=True)
        f_local_a = model2.local_proj(roi_a)
        f_local_b = model2.local_proj(roi_b)
    all_ok &= check("f_local DOES differ between boxes (real box-dependent signal pre-zero-mult)",
                     not torch.allclose(f_local_a, f_local_b))

    # 4. zero-init columns are actually zero (not just numerically close)
    all_ok &= check("fusion_head local columns are exactly zero",
                     torch.equal(model2.fusion_head.fc.weight[:, GLOBAL_DIM:],
                                 torch.zeros_like(model2.fusion_head.fc.weight[:, GLOBAL_DIM:])))

    # 5. roi_align sanity: full-image box vs a tiny corner box give different
    #    pooled features (direct roi_align check, independent of the model)
    feat = torch.randn(1, 320, 14, 14)
    full_box = [torch.tensor([[0.0, 0.0, 224.0, 224.0]])]
    corner_box = [torch.tensor([[0.0, 0.0, 32.0, 32.0]])]
    roi_full = roi_align(feat, full_box, output_size=(3, 3), spatial_scale=14 / 224, aligned=True)
    roi_corner = roi_align(feat, corner_box, output_size=(3, 3), spatial_scale=14 / 224, aligned=True)
    all_ok &= check("roi_align output differs for full-image vs corner box",
                     not torch.allclose(roi_full, roi_corner))

    # 6. freeze_backbone / unfreeze_last_stage toggle requires_grad correctly
    model3 = build_model()
    freeze_backbone(model3)
    all_ok &= check("freeze_backbone: all backbone params frozen",
                     all(not p.requires_grad for p in model3.backbone.parameters()))
    all_ok &= check("freeze_backbone: local_proj/fusion_head still trainable",
                     all(p.requires_grad for p in model3.local_proj.parameters())
                     and all(p.requires_grad for p in model3.fusion_head.parameters()))
    unfreeze_last_stage(model3)
    n_unfrozen = sum(p.requires_grad for p in model3.backbone.parameters())
    all_ok &= check(f"unfreeze_last_stage: {n_unfrozen} backbone params now trainable (>0)", n_unfrozen > 0)

    print(f"\nparams: {count_params_m(model):.3f}M total")
    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
