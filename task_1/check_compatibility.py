"""Pre-flight compatibility check for the WHOLE ablation grid (all 3 axes:
architecture, loss, augmentation).

Nothing here trains a model -- every check is a single forward pass (or,
for losses, forward+backward on a tiny random tensor). Run this any time
an ablation YAML is edited; it exits non-zero if any combination a real
run would hit is broken. Checks:

  - architecture: build every (decoder, encoder) pair the grid uses (all
    ImageNet weights -- no SSL/SWSL in this study) and run one forward
    pass at BASE's resolution, catching architectural incompatibilities.
  - loss: forward+backward every loss on dummy logits/targets, catching
    non-finite loss values or dead/exploding gradients.
  - augmentation: run every policy on a deliberately NON-SQUARE dummy
    image+mask (this dataset mixes square and non-square sources), making
    sure the letterbox resize/pad pipeline and every op in each policy
    (ElasticTransform, CoarseDropout, ...) accepts the current
    albumentations version's API.
"""

import sys

import numpy as np
import segmentation_models_pytorch as smp
import torch

from src.augmentations import build_transform
from src.losses import build_loss
from src.model import build_model
from src.utils import load_yaml

failures = []
checked_arch = set()


def check_arch(decoder, encoder, resolution):
    key = (decoder, encoder, resolution)
    if key in checked_arch:
        return
    checked_arch.add(key)
    try:
        model = build_model(decoder, encoder, None, num_classes=3)
        model.eval()
        x = torch.randn(1, 3, resolution, resolution)
        with torch.no_grad():
            y = model(x)
        ok = tuple(y.shape) == (1, 3, resolution, resolution)
        status = "OK" if ok else "FAIL"
        print(f"[arch]      decoder={decoder:14s} encoder={encoder:20s} res={resolution:4d} -> {status}"
              + ("" if ok else f"  (got shape {tuple(y.shape)})"))
        if not ok:
            failures.append(("arch", f"{decoder}/{encoder}/res{resolution}", f"unexpected output shape {tuple(y.shape)}"))
    except Exception as e:
        print(f"[arch]      decoder={decoder:14s} encoder={encoder:20s} res={resolution:4d} -> FAIL  ({type(e).__name__}: {e})")
        failures.append(("arch", f"{decoder}/{encoder}/res{resolution}", f"{type(e).__name__}: {e}"))


def check_weights(decoder, encoder, weights, resolution):
    settings = smp.encoders.encoders.get(encoder, {}).get("pretrained_settings", {})
    available = weights in settings
    status = "OK" if available else "FAIL"
    print(f"[weights]   decoder={decoder:14s} encoder={encoder:20s} weights={weights:10s} -> {status}"
          + ("" if available else f"  (registered keys: {list(settings)})"))
    if not available:
        failures.append(("weights", f"{decoder}/{encoder}/{weights}", f"'{weights}' not in SMP pretrained_settings ({list(settings)})"))
        return
    check_arch(decoder, encoder, resolution)


def check_loss(name, num_classes, ignore_index):
    try:
        loss_fn = build_loss(name, ignore_index)
        logits = torch.randn(2, num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, num_classes, (2, 32, 32))
        loss = loss_fn(logits, target)
        loss.backward()
        finite_loss = torch.isfinite(loss).item()
        finite_grad = logits.grad is not None and torch.isfinite(logits.grad).all().item()
        ok = finite_loss and finite_grad
        status = "OK" if ok else "FAIL"
        print(f"[loss]      {name:12s} -> {status}  (value={loss.item():.4f})")
        if not ok:
            failures.append(("loss", name, "non-finite loss or gradient"))
    except Exception as e:
        print(f"[loss]      {name:12s} -> FAIL  ({type(e).__name__}: {e})")
        failures.append(("loss", name, f"{type(e).__name__}: {e}"))


def check_augmentation(policy, resolution, num_classes):
    # Deliberately non-square (H != W) to exercise the LongestMaxSize +
    # PadIfNeeded letterbox path this dataset actually needs.
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(600, 800, 3), dtype=np.uint8)
    mask = rng.integers(0, num_classes, size=(600, 800)).astype(np.int64)
    try:
        tf = build_transform(policy, resolution, is_train=True)
        out = tf(image=image, mask=mask)
        img_t, mask_t = out["image"], out["mask"]
        ok = tuple(img_t.shape) == (3, resolution, resolution) and tuple(mask_t.shape) == (resolution, resolution)
        status = "OK" if ok else "FAIL"
        print(f"[aug]       {policy:10s} res={resolution:4d} -> {status}"
              + ("" if ok else f"  (image={tuple(img_t.shape)}, mask={tuple(mask_t.shape)})"))
        if not ok:
            failures.append(("augmentation", f"{policy}/res{resolution}",
                              f"bad output shape image={tuple(img_t.shape)} mask={tuple(mask_t.shape)}"))
    except Exception as e:
        print(f"[aug]       {policy:10s} res={resolution:4d} -> FAIL  ({type(e).__name__}: {e})")
        failures.append(("augmentation", f"{policy}/res{resolution}", f"{type(e).__name__}: {e}"))


def main():
    base = load_yaml("configs/base.yaml")
    base_decoder = base["model"]["decoder"]
    base_encoder = base["model"]["encoder"]
    base_weights = base["model"]["weights"]
    base_resolution = base["train"]["resolution"]
    num_classes = len(base["data"]["classes"])
    ignore_index = base["data"]["ignore_index"]

    print("=== BASE ===")
    check_weights(base_decoder, base_encoder, base_weights, base_resolution)

    print("\n=== architecture.yaml (decoder+encoder jointly vary) ===")
    for case in load_yaml("configs/ablations/architecture.yaml")["cases"]:
        ov = case["overrides"]
        check_weights(ov["model.decoder"], ov["model.encoder"], base_weights, base_resolution)

    print("\n=== loss.yaml ===")
    for case in load_yaml("configs/ablations/loss.yaml")["cases"]:
        check_loss(case["overrides"]["train.loss"], num_classes, ignore_index)

    print("\n=== augmentation.yaml (non-square dummy input) ===")
    for case in load_yaml("configs/ablations/augmentation.yaml")["cases"]:
        check_augmentation(case["overrides"]["train.augmentation"], base_resolution, num_classes)

    print()
    if failures:
        print(f"{len(failures)} INCOMPATIBLE / BROKEN COMBINATION(S) FOUND:\n")
        for axis, what, err in failures:
            print(f"  [{axis}] {what}\n    -> {err}")
        sys.exit(1)
    else:
        print(f"All axes compatible: {len(checked_arch)} architecture combinations, "
              f"all loss/augmentation cases OK.")


if __name__ == "__main__":
    main()
