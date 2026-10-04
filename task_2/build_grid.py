#!/usr/bin/env python3
"""Resolves configs/grid.yaml into configs/resolved_grid.json.

Cross-product of encoders x heads (13 encoders x 2 heads = 26 configs).
Dry-runs each with a dummy batch to fail loudly on a typo/incompatible
encoder and records each config's parameter count. Must be run once before
run_ablation.py / run_efficiency.py.
"""
import json
import sys
from pathlib import Path

import torch

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR))

from src.model import build_model, count_params_m  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def main():
    grid_cfg = load_yaml(TASK_DIR / "configs" / "grid.yaml")
    base_cfg = load_yaml(TASK_DIR / "configs" / "base.yaml")
    weights = grid_cfg["default_weights"]
    num_classes = len(base_cfg["classes"])

    resolved = []
    for encoder_entry in grid_cfg["encoders"]:
        encoder = encoder_entry["name"]
        for head in grid_cfg["heads"]:
            model = build_model(encoder, weights, head, num_classes=num_classes)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(1, 3, base_cfg["image_size"], base_cfg["image_size"]))
            expected_dim = num_classes if head == "softmax" else num_classes - 1
            assert out.shape == (1, expected_dim), f"unexpected output shape {tuple(out.shape)}"

            resolved.append({
                "config_id": f"{encoder}__{head}",
                "encoder": encoder,
                "head": head,
                "weights": weights,
                "params_m": round(count_params_m(model), 3),
            })

    out_path = TASK_DIR / "configs" / "resolved_grid.json"
    out_path.write_text(json.dumps(resolved, indent=2))
    print(f"Resolved {len(resolved)} configs -> {out_path}")
    for c in resolved:
        print(f"  - {c['config_id']} ({c['params_m']}M params)")


if __name__ == "__main__":
    main()
