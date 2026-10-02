#!/usr/bin/env python3
"""Measures params(M) and single-image inference latency (ms) for each of
the 10 selected configs in configs/selected_configs.json. Run after
build_selected_configs.py.
"""
import csv
import json
import sys
from pathlib import Path

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR))

from src.efficiency import count_params_m, measure_latency_ms  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402


def main():
    base_cfg = load_yaml(TASK_DIR / "configs" / "base.yaml")
    resolved_path = TASK_DIR / "configs" / "selected_configs.json"
    if not resolved_path.exists():
        raise SystemExit("configs/selected_configs.json not found — run build_selected_configs.py first")
    grid = json.loads(resolved_path.read_text())

    device = get_device()
    num_classes = len(base_cfg["classes"])

    out_path = TASK_DIR / "results" / "efficiency.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["config_id", "architecture", "encoder", "weights", "params_m", "latency_ms"]
        )
        writer.writeheader()
        for cfg in grid:
            model = build_model(cfg["architecture"], cfg["encoder"], None, num_classes=num_classes)
            params_m = count_params_m(model)
            latency_ms = measure_latency_ms(model, device, image_size=base_cfg["image_size"])
            row = {
                "config_id": cfg["config_id"], "architecture": cfg["architecture"],
                "encoder": cfg["encoder"], "weights": cfg["weights"],
                "params_m": round(params_m, 3), "latency_ms": round(latency_ms, 2),
            }
            writer.writerow(row)
            print(row, flush=True)


if __name__ == "__main__":
    main()
