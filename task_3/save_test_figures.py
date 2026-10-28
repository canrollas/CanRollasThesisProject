#!/usr/bin/env python3
"""One Matplotlib PNG per test case: RGB, labels, logits, probabilities and CAM.

Defaults to mistakes from the three original held-out MiT-B1+CORN folds.
--all saves correct cases too. No training, aggregate analysis or HTML report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mit-b1-corn-matplotlib")

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from corn_explain import corn_outputs, extract_explanations
from error_example_figures import render_test_example
from src.dataset import StageDataset
from src.model import build_model
from src.splits import build_folds

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/wound-stage-classification")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "task_3/results/task_3_outputs/checkpoints")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "more_experiments/reports/figures/mit_b1_corn_test_cases")
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--all", action="store_true", help="Save every test case, including correct predictions")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    fold_ids = [int(value) for value in args.folds.split(",")]
    if not fold_ids or len(set(fold_ids)) != len(fold_ids) or any(value not in range(3) for value in fold_ids):
        parser.error("--folds must select unique values from 0,1,2")
    protocol_path = ROOT / "task_3/configs/mit_b1_corn_baseline_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    folds = build_folds(args.data_root, num_folds=3, val_fraction=.1, seed=42)
    for fold_id in fold_ids:
        expected = protocol["folds"][str(fold_id)]
        split_sha = hashlib.sha256(json.dumps(folds[fold_id], sort_keys=True).encode()).hexdigest()
        checkpoint = args.checkpoint_dir / f"mit_b1__corn_fold{fold_id}.pt"
        if split_sha != expected["split_sha256"] or sha256(checkpoint) != expected["baseline_sha256"]:
            raise ValueError(f"Fold {fold_id} must use the original test split and RGB checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold_id in fold_ids:
        model = build_model("mit_b1", None, "corn", num_classes=4)
        checkpoint = args.checkpoint_dir / f"mit_b1__corn_fold{fold_id}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.to(args.device).eval()
        samples = folds[fold_id]["test"]
        loader = DataLoader(StageDataset(samples, args.data_root, 224, transform=None),
                            batch_size=1, shuffle=False, num_workers=0)
        print(f"Fold {fold_id}: {len(samples)} test images", flush=True)
        saved = 0
        for index, (images, targets) in enumerate(loader):
            relative, true = samples[index]
            images = images.to(args.device)
            with torch.no_grad():
                outputs = corn_outputs(model(images))
            pred = int(outputs["decoded"][0])
            probability = outputs["class_probabilities"][0].cpu().numpy()
            row = {"path": relative, "fold": fold_id, "true_label": true, "pred_label": pred,
                   "correct": pred == true, "prob_argmax": int(outputs["probability_argmax"][0]),
                   "predicted_probability": float(probability[pred]), "figure_path": ""}
            for prefix, name in (("logit", "logits"), ("conditional", "conditional_probabilities"),
                                 ("survival", "survival_probabilities"), ("prob", "class_probabilities")):
                row.update({f"{prefix}_{j}": float(value) for j, value in enumerate(outputs[name][0].cpu())})
            if args.all or pred != true:
                explanation = extract_explanations(model, images, targets=targets.to(args.device),
                                                   include_thresholds=False)
                if int(explanation["decoded"][0]) != pred:
                    raise AssertionError("Prediction changed during explanation")
                cams = {target: {layer: values["normalized"][0].numpy()
                                 for layer, values in layers.items()}
                        for target, layers in explanation["cams"].items()}
                with Image.open(args.data_root / relative) as source:
                    rgb = np.asarray(source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
                destination = args.output_dir / f"fold_{fold_id}" / Path(relative).with_suffix(".png")
                render_test_example(rgb, row, cams, destination)
                row["figure_path"] = destination.relative_to(args.output_dir).as_posix()
                saved += 1
            rows.append(row)
            if (index + 1) % 25 == 0 or index + 1 == len(samples):
                print(f"  {index+1}/{len(samples)} · {saved} PNG saved", flush=True)
        del model
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Figures: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
