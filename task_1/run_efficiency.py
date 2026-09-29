import argparse
import csv
import json
from pathlib import Path

import torch

from src.efficiency import count_params, measure_latency
from src.model import build_model


def main():
    parser = argparse.ArgumentParser(description="Params/latency benchmark for all trained checkpoints")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    # Latency/params depend only on architecture, not on trained weights or
    # data split, so one fold per case (fold0) is enough.
    config_paths = sorted(Path(args.checkpoint_root).rglob("*_fold0_config.json"))
    if not config_paths:
        raise SystemExit(f"No *_fold0_config.json files found under {args.checkpoint_root}")

    rows = []
    for config_path in config_paths:
        config = json.load(open(config_path))
        num_classes = len(config["data"]["classes"])
        model = build_model(config["model"]["decoder"], config["model"]["encoder"], config["model"]["weights"], num_classes)
        model.to(args.device)

        latency_ms = measure_latency(model, config["train"]["resolution"], args.device, batch_size=args.batch_size)
        row = {
            "config": str(config_path),
            "decoder": config["model"]["decoder"],
            "encoder": config["model"]["encoder"],
            "weights": config["model"]["weights"],
            "resolution": config["train"]["resolution"],
            "params_m": count_params(model) / 1e6,
            "latency_ms": latency_ms,
        }
        rows.append(row)
        print(row)

    out_csv = Path(args.checkpoint_root) / "efficiency_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
