#!/usr/bin/env python3
"""Resolves configs/selected.yaml into configs/selected_configs.json.

Unlike task_1's check_compatibility.py (which dry-runs a full architecture x
encoder grid to discover which pairs are valid), this task starts from a
hand-picked top-10 list (the best configs from the paper's 37-config Table 5
benchmark) that's already known to be SMP-compatible. This script still
dry-runs each pick with a dummy batch to fail loudly on a typo, and records
each config's parameter count. Must be run once before run_ablation.py /
run_efficiency.py.
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
    selected = load_yaml(TASK_DIR / "configs" / "selected.yaml")["configs"]

    resolved = []
    for entry in selected:
        arch, encoder, weights = entry["architecture"], entry["encoder"], entry["weights"]
        model = build_model(arch, encoder, None, num_classes=3)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 256, 256))
        assert out.shape == (1, 3, 256, 256), f"unexpected output shape {tuple(out.shape)}"

        resolved.append({
            "config_id": f"{arch}__{encoder}__{weights}",
            "architecture": arch,
            "encoder": encoder,
            "weights": weights,
            "params_m": round(count_params_m(model), 3),
            "paper_miou": entry.get("paper_miou"),
        })

    out_path = TASK_DIR / "configs" / "selected_configs.json"
    out_path.write_text(json.dumps(resolved, indent=2))
    print(f"Resolved {len(resolved)} configs -> {out_path}")
    for c in resolved:
        print(f"  - {c['config_id']} ({c['params_m']}M params, paper mIoU={c['paper_miou']})")


if __name__ == "__main__":
    main()
