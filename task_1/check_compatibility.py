"""Pre-flight compatibility check for the whole ablation grid.

Builds every (decoder, encoder, weights) combination that any study in
configs/ablations/*.yaml will actually instantiate, plus every resolution
BASE's model will be trained at, and runs one forward pass on a random
tensor (no dataset, no training) to catch:
  - decoder/encoder architectural incompatibilities (shape mismatches,
    unsupported encoder depth/dilation for that decoder, etc.)
  - weights keys (ssl/swsl) not registered for a given encoder in SMP

Uses encoder_weights=None for the architecture/resolution checks so no
network access or download is required; weights availability is checked
against SMP's pretrained_settings metadata, also without downloading.

Run this any time an ablation YAML is edited -- exits non-zero if any
combination the grid would actually run is broken.
"""

import sys

import segmentation_models_pytorch as smp
import torch

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
        print(f"[arch]    decoder={decoder:14s} encoder={encoder:20s} res={resolution:4d} -> {status}"
              + ("" if ok else f"  (got shape {tuple(y.shape)})"))
        if not ok:
            failures.append(("arch", decoder, encoder, None, resolution, f"unexpected output shape {tuple(y.shape)}"))
    except Exception as e:
        print(f"[arch]    decoder={decoder:14s} encoder={encoder:20s} res={resolution:4d} -> FAIL  ({type(e).__name__}: {e})")
        failures.append(("arch", decoder, encoder, None, resolution, f"{type(e).__name__}: {e}"))


def check_weights(decoder, encoder, weights, resolution):
    settings = smp.encoders.encoders.get(encoder, {}).get("pretrained_settings", {})
    available = weights in settings
    status = "OK" if available else "FAIL"
    print(f"[weights] decoder={decoder:14s} encoder={encoder:20s} weights={weights:10s} -> {status}"
          + ("" if available else f"  (registered keys: {list(settings)})"))
    if not available:
        failures.append(("weights", decoder, encoder, weights, resolution, f"'{weights}' not in SMP pretrained_settings ({list(settings)})"))
        return
    check_arch(decoder, encoder, resolution)


def main():
    base = load_yaml("configs/base.yaml")
    base_decoder = base["model"]["decoder"]
    base_encoder = base["model"]["encoder"]
    base_weights = base["model"]["weights"]
    base_resolution = base["train"]["resolution"]

    print("=== BASE ===")
    check_weights(base_decoder, base_encoder, base_weights, base_resolution)

    print("\n=== backbone.yaml (decoder fixed) ===")
    for case in load_yaml("configs/ablations/backbone.yaml")["cases"]:
        encoder = case["overrides"]["model.encoder"]
        check_weights(base_decoder, encoder, base_weights, base_resolution)

    print("\n=== decoder.yaml (encoder fixed) ===")
    for case in load_yaml("configs/ablations/decoder.yaml")["cases"]:
        decoder = case["overrides"]["model.decoder"]
        check_weights(decoder, base_encoder, base_weights, base_resolution)

    print("\n=== weights.yaml (decoder+encoder fixed to resnet50) ===")
    for case in load_yaml("configs/ablations/weights.yaml")["cases"]:
        ov = case["overrides"]
        check_weights(ov["model.decoder"], ov["model.encoder"], ov["model.weights"], base_resolution)

    print("\n=== resolution.yaml (decoder+encoder = BASE) ===")
    for case in load_yaml("configs/ablations/resolution.yaml")["cases"]:
        resolution = case["overrides"]["train.resolution"]
        check_weights(base_decoder, base_encoder, base_weights, resolution)

    print()
    if failures:
        print(f"{len(failures)} INCOMPATIBLE COMBINATION(S) FOUND:\n")
        for kind, decoder, encoder, weights, resolution, err in failures:
            print(f"  [{kind}] decoder={decoder} encoder={encoder} weights={weights} res={resolution}\n    -> {err}")
        sys.exit(1)
    else:
        print(f"All {len(checked_arch)} architecture combinations and every weights case are compatible.")


if __name__ == "__main__":
    main()
