#!/usr/bin/env python3
"""Architecture-only sanity check for exp13 -- NO pretrained weights
downloaded (pretrained=False everywhere here; DINOv2 ViT-L's real
weights are ~1.2GB, fetched only on Colab, not locally). Verifies: the
model builds, forward pass shape is correct at 224x224 (patch14 -> exact
16x16 grid), frozen mode truly blocks backbone gradients, and finetune
mode truly doesn't.
"""
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from model import build_model, count_params_m, count_trainable_params_m  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    torch.manual_seed(0)
    all_ok = True

    model = build_model("frozen", num_classes=4, img_size=224, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    all_ok &= check("CORN logits shape == [2, 3] (num_classes-1)", tuple(logits.shape) == (2, 3))
    all_ok &= check("out_dim == 1024 (ViT-L hidden size)", model.out_dim == 1024)

    params_m = count_params_m(model)
    trainable_m_frozen = count_trainable_params_m(model)
    all_ok &= check(f"frozen mode: ~303M total params (got {params_m:.1f}M)", 295 < params_m < 310)
    all_ok &= check(
        f"frozen mode: only CornHead trains (got {trainable_m_frozen:.4f}M trainable, expect ~0.003M)",
        trainable_m_frozen < 0.01,
    )

    # frozen mode: a backward pass must leave the encoder's grads untouched
    model.zero_grad()
    loss = logits.sum()
    loss.backward()
    encoder_grad_is_none = all(p.grad is None for p in model.encoder.parameters())
    all_ok &= check("frozen mode: encoder.parameters() got NO gradient after backward()", encoder_grad_is_none)

    # finetune mode: encoder must be trainable
    model_ft = build_model("finetune", num_classes=4, img_size=224, pretrained=False)
    trainable_m_ft = count_trainable_params_m(model_ft)
    all_ok &= check(
        f"finetune mode: ~303M trainable (got {trainable_m_ft:.1f}M, same as total)",
        abs(trainable_m_ft - count_params_m(model_ft)) < 0.01,
    )

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
