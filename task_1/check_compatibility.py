#!/usr/bin/env python3
"""Pre-flight grid resolver.

Dry-runs every (architecture, encoder) pair from configs/grid.yaml with a
256x256 dummy batch (encoder_weights=None, so no network access is needed),
drops whatever SMP/our SegFormer head can't build, then expands the survivors
with their pretrained-weight variants and writes configs/resolved_grid.json.
Must be run once before run_ablation.py / run_efficiency.py.
"""
import json
import sys
from pathlib import Path

import torch

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR))

from src.model import build_model, build_segformer, count_params_m  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def try_build(builder, *args, **kwargs):
    try:
        model = builder(*args, **kwargs)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 256, 256))
        assert out.shape == (1, 3, 256, 256), f"unexpected output shape {tuple(out.shape)}"
        return model, None
    except Exception as e:  # noqa: BLE001 - intentionally broad, this is a probe
        return None, f"{type(e).__name__}: {e}"


def main():
    grid = load_yaml(TASK_DIR / "configs" / "grid.yaml")
    architectures = grid["architectures"]
    encoders = grid["encoders"]
    default_weights = grid["default_weights"]
    extra = grid["extra_weights"]
    segformer_cfg = grid["segformer"]

    resolved = []
    skipped = []

    ok = {}
    for arch in architectures:
        for enc in encoders:
            enc_name = enc["name"]
            _, err = try_build(build_model, arch, enc_name, None, num_classes=3)
            ok[(arch, enc_name)] = err is None
            if err is not None:
                skipped.append({"architecture": arch, "encoder": enc_name, "reason": err})

    for arch in architectures:
        for enc in encoders:
            enc_name = enc["name"]
            if not ok[(arch, enc_name)]:
                continue
            weight_variants = [default_weights]
            if enc_name == extra["encoder"]:
                weight_variants += extra["weights"]
            for w in weight_variants:
                model = build_model(arch, enc_name, None, num_classes=3)
                resolved.append({
                    "config_id": f"{arch}__{enc_name}__{w}",
                    "architecture": arch,
                    "encoder": enc_name,
                    "weights": w,
                    "params_m": round(count_params_m(model), 3),
                })

    for enc_name in segformer_cfg["encoders"]:
        model, err = try_build(
            build_segformer, enc_name, None,
            num_classes=3, embed_dim=segformer_cfg.get("decoder_embed_dim", 256),
        )
        if err is not None:
            skipped.append({"architecture": "segformer", "encoder": enc_name, "reason": err})
            continue
        resolved.append({
            "config_id": f"segformer__{enc_name}__{segformer_cfg['weights']}",
            "architecture": "segformer",
            "encoder": enc_name,
            "weights": segformer_cfg["weights"],
            "params_m": round(count_params_m(model), 3),
        })

    out_path = TASK_DIR / "configs" / "resolved_grid.json"
    out_path.write_text(json.dumps(resolved, indent=2))

    print(f"Resolved {len(resolved)} valid configs -> {out_path}")
    if skipped:
        print(f"Skipped {len(skipped)} incompatible (architecture, encoder) pairs:")
        for s in skipped:
            print(f"  - {s['architecture']} + {s['encoder']}: {s['reason']}")


if __name__ == "__main__":
    main()
