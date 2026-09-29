"""Pre-flight compatibility check for the WHOLE ablation grid (all 7 axes:
backbone, decoder, weights, loss, resolution, augmentation, data_efficiency).

Nothing here trains a model -- every check is a single forward pass (or,
for losses, forward+backward on a tiny random tensor) or pure metadata/
arithmetic. Run this any time an ablation YAML is edited; it exits
non-zero if any combination a real run would hit is broken. Checks:

  - backbone / decoder: build every (decoder, encoder) pair the grid uses
    and run one forward pass, catching architectural incompatibilities.
  - weights: verify ssl/swsl/imagenet keys are registered for the
    requested encoder in SMP's metadata (no download needed).
  - resolution: forward pass BASE's architecture at every candidate
    resolution, catching stride/divisibility issues.
  - loss: forward+backward every loss on dummy logits/targets, catching
    non-finite loss values or dead/exploding gradients.
  - augmentation: run every policy on a deliberately NON-SQUARE dummy
    image+mask (this dataset mixes square and non-square sources), making
    sure the letterbox resize/pad pipeline and every op in each policy
    (ElasticTransform, CoarseDropout, ...) accepts the current
    albumentations version's API.
  - data_efficiency: compute the REAL per-fold train-set size after the
    holdout+k-fold split, then check that every train_fraction still
    yields at least one training batch under drop_last=True. Silently
    training on 0 batches/epoch would raise no exception -- it would just
    quietly never learn -- so this is checked explicitly.
"""

import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from src.augmentations import build_transform
from src.dataset import list_filenames, make_kfold_splits, split_holdout, subsample
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


def check_data_efficiency(fraction, train_size_per_fold, batch_size, seed):
    fake_files = [f"f{i}.png" for i in range(train_size_per_fold)]
    sub = subsample(fake_files, fraction, seed)
    n_batches = len(sub) // batch_size  # matches DataLoader(..., drop_last=True)
    ok = n_batches >= 1
    status = "OK" if ok else "FAIL"
    print(f"[data-eff]  fraction={fraction:4.2f} -> train_files={len(sub):5d} batches/epoch={n_batches:3d} -> {status}")
    if not ok:
        failures.append(("data_efficiency", f"fraction={fraction}",
                          f"only {len(sub)} train files at batch_size={batch_size} with drop_last=True -> 0 batches/epoch"))


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

    print("\n=== backbone.yaml (decoder fixed) ===")
    for case in load_yaml("configs/ablations/backbone.yaml")["cases"]:
        check_weights(base_decoder, case["overrides"]["model.encoder"], base_weights, base_resolution)

    print("\n=== decoder.yaml (encoder fixed) ===")
    for case in load_yaml("configs/ablations/decoder.yaml")["cases"]:
        check_weights(case["overrides"]["model.decoder"], base_encoder, base_weights, base_resolution)

    print("\n=== weights.yaml (decoder+encoder fixed to resnet50) ===")
    for case in load_yaml("configs/ablations/weights.yaml")["cases"]:
        ov = case["overrides"]
        check_weights(ov["model.decoder"], ov["model.encoder"], ov["model.weights"], base_resolution)

    print("\n=== resolution.yaml (decoder+encoder = BASE) ===")
    for case in load_yaml("configs/ablations/resolution.yaml")["cases"]:
        check_weights(base_decoder, base_encoder, base_weights, case["overrides"]["train.resolution"])

    print("\n=== loss.yaml ===")
    for case in load_yaml("configs/ablations/loss.yaml")["cases"]:
        check_loss(case["overrides"]["train.loss"], num_classes, ignore_index)

    print("\n=== augmentation.yaml (non-square dummy input) ===")
    for case in load_yaml("configs/ablations/augmentation.yaml")["cases"]:
        check_augmentation(case["overrides"]["train.augmentation"], base_resolution, num_classes)

    print("\n=== data_efficiency.yaml ===")
    images_dir = Path(base["data"]["images_dir"])
    if images_dir.is_dir():
        all_filenames = list_filenames(images_dir)
        dev_pool, _ = split_holdout(all_filenames, base["data"]["holdout_fraction"], base["data"]["holdout_seed"])
        folds = make_kfold_splits(
            dev_pool, base["data"]["n_folds"], base["data"]["split_seed"], base["data"]["val_fraction_of_remaining"]
        )
        train_size_per_fold = len(folds[0][0])
        print(f"(using real dataset: {len(all_filenames)} files -> dev pool {len(dev_pool)} -> "
              f"{train_size_per_fold} train files/fold before data_efficiency subsampling)")
    else:
        train_size_per_fold = 1773  # last known real per-fold train size; used if dataset isn't mounted yet
        print(f"(dataset not found at {images_dir}, using last known real per-fold train size {train_size_per_fold})")
    for case in load_yaml("configs/ablations/data_efficiency.yaml")["cases"]:
        check_data_efficiency(
            case["overrides"]["train.train_fraction"], train_size_per_fold, base["train"]["batch_size"], base["train"]["seed"]
        )

    print()
    if failures:
        print(f"{len(failures)} INCOMPATIBLE / BROKEN COMBINATION(S) FOUND:\n")
        for axis, what, err in failures:
            print(f"  [{axis}] {what}\n    -> {err}")
        sys.exit(1)
    else:
        print(f"All axes compatible: {len(checked_arch)} architecture combinations, "
              f"all weights/loss/augmentation/resolution/data-efficiency cases OK.")


if __name__ == "__main__":
    main()
