#!/usr/bin/env python3
"""Evaluate a fixed Stage-1 gate followed by genuine Stage-2/3/4 specialists.

The gate is the existing four-class RGB CORN model for the held-out fold.
Its decoded Stage-1 decision is final. Other images are sent to each newly
trained three-class specialist, whose decoded index is shifted by one.
True stage labels are used only for evaluation, never for routing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))
sys.path.insert(0, str(THIS_DIR))

from src.dataset import CLASS_ORDER  # noqa: E402
from src.metrics import compute_metrics, predict_labels  # noqa: E402
from src.model import build_model as build_gate  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device  # noqa: E402
from data import PairedStageDataset, dataset_fingerprint  # noqa: E402
from model import build_variant  # noqa: E402

VARIANTS = ("rgb", "rgb_tissue")
PREDICTION_COLUMNS = {
    "four_class_rgb_baseline": "baseline_pred",
    "cascade_rgb": "cascade_rgb_pred",
    "cascade_rgb_tissue": "cascade_rgb_tissue_pred",
}


def parse_folds(value):
    try:
        folds = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("folds must be comma-separated integers") from exc
    if not folds or len(set(folds)) != len(folds) or any(i not in range(3) for i in folds):
        raise argparse.ArgumentTypeError("select distinct folds from 0,1,2")
    return folds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gate-checkpoints-dir", required=True, type=Path)
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("0,1,2"))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0:
        parser.error("batch must be positive and workers must be nonnegative")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_training_protocol(output_dir):
    path = output_dir / "results" / "run_protocol.json"
    if not path.is_file():
        raise FileNotFoundError(f"Training protocol missing: {path}. Run train.py first.")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    config = protocol.get("config", protocol)
    if config.get("classes") != ["stage_2", "stage_3", "stage_4"]:
        raise ValueError("Training protocol must describe exactly Stage 2/3/4, in that order")
    if config.get("head", "corn") != "corn" or config.get("encoder", "mit_b1") != "mit_b1":
        raise ValueError("Cascade evaluation expects the MiT-B1 + CORN training protocol")
    if protocol.get("smoke", config.get("smoke", False)):
        raise ValueError("Smoke checkpoints cannot be reported as the full cascade experiment")
    for key in ("image_size", "num_folds", "val_fraction", "seed"):
        if key not in config:
            raise ValueError(f"Training protocol is missing {key}")
    if config["num_folds"] != 3:
        raise ValueError("The existing gate checkpoints require the original three-fold protocol")
    return protocol, config


def verify_training_inputs(output_dir, data_root, folds, protocol):
    if dataset_fingerprint(data_root) != protocol.get("dataset_fingerprint"):
        raise ValueError("Dataset differs from the specialist training run; use the same input dataset")
    manifest_path = output_dir / "results" / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training split manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("original_classes") != CLASS_ORDER:
        raise ValueError("Training manifest must preserve the original four-class label order")
    saved_folds = {item["fold"]: item for item in manifest["folds"]}
    for fold_idx, original in enumerate(folds):
        if fold_idx not in saved_folds:
            raise ValueError(f"Training manifest is missing fold {fold_idx}")
        for split in ("train", "val", "test"):
            expected = [(path, label - 1) for path, label in original[split] if label > 0]
            actual = [(item["path"], item["label"]) for item in saved_folds[fold_idx][split]]
            if actual != expected:
                raise ValueError(f"Training membership changed for fold {fold_idx} {split}")


def load_gate_protocol(data_root, config, folds):
    """Check exact historical membership without requiring the RGB archive.

    The standard eight-channel dataset's metadata retains each original
    image filename, including its extension. Restoring those paths lets us
    verify the same historical split hashes used by Task 3 explanations.
    """
    path = REPO_ROOT / "task_3" / "configs" / "mit_b1_corn_baseline_protocol.json"
    historical = json.loads(path.read_text(encoding="utf-8"))
    for key, historical_key in (
        ("seed", "split_seed"), ("num_folds", "num_folds"),
        ("val_fraction", "val_fraction"), ("image_size", "image_size"),
    ):
        if config[key] != historical[historical_key]:
            raise ValueError(f"Specialist {key} differs from the historical fixed gate protocol")
    metadata_path = data_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Original-filename metadata required to verify the gate split: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        original_names = {}
        for item in csv.DictReader(handle):
            key = (item["stage"], item["npy_file"])
            if key in original_names:
                raise ValueError(f"Duplicate source image in metadata: {key}")
            original_names[key] = item["filename"]
    test_paths = [path for fold in folds for path, _ in fold["test"]]
    if len(test_paths) != historical["n_images"] or len(set(test_paths)) != historical["n_images"]:
        raise ValueError("Gate test folds must partition all 1,091 historical images exactly once")
    for fold_idx, fold in enumerate(folds):
        restored = {
            split: [(str(Path(path).parent / original_names[(Path(path).parent.name, Path(path).name)]), label)
                    for path, label in samples]
            for split, samples in fold.items()
        }
        digest = hashlib.sha256(json.dumps(restored, sort_keys=True).encode()).hexdigest()
        if digest != historical["folds"][str(fold_idx)]["split_sha256"]:
            raise ValueError(f"Fold {fold_idx} membership differs from the historical gate checkpoint")
    return historical


def build_reports(predictions):
    metrics, per_class, confusion = [], [], []
    fold_values = sorted({row["fold"] for row in predictions})
    for scope in [*fold_values, "pooled"]:
        selected = predictions if scope == "pooled" else [r for r in predictions if r["fold"] == scope]
        true = np.asarray([row["true_index"] for row in selected], dtype=np.int64)
        for model_name, prediction_key in PREDICTION_COLUMNS.items():
            pred = np.asarray([row[prediction_key] for row in selected], dtype=np.int64)
            stats = compute_metrics(true, pred, num_classes=4)
            metrics.append({
                "model": model_name, "fold": scope, "n": len(true),
                **{key: float(stats[key]) for key in ("accuracy", "macro_f1", "mae", "qwk")},
            })
            precision, recall, f1, support = precision_recall_fscore_support(
                true, pred, labels=list(range(4)), zero_division=0,
            )
            for label, name in enumerate(CLASS_ORDER):
                per_class.append({
                    "model": model_name, "fold": scope, "class": name,
                    "precision": float(precision[label]), "recall": float(recall[label]),
                    "f1": float(f1[label]), "support": int(support[label]),
                })
                for predicted_label, predicted_name in enumerate(CLASS_ORDER):
                    confusion.append({
                        "model": model_name, "fold": scope,
                        "true_class": name, "predicted_class": predicted_name,
                        "count": int(stats["confusion"][label, predicted_label]),
                    })
    return metrics, per_class, confusion


@torch.no_grad()
def evaluate_fold(fold_idx, samples, data_root, image_size, checkpoint_paths, device, batch, workers):
    gate = build_gate("mit_b1", None, "corn", num_classes=4)
    gate.load_state_dict(torch.load(checkpoint_paths["gate"], map_location="cpu", weights_only=True))
    gate.to(device).eval()
    specialists = {}
    for variant in VARIANTS:
        model = build_variant(variant, weights=None, encoder="mit_b1")
        model.load_state_dict(torch.load(checkpoint_paths[variant], map_location="cpu", weights_only=True))
        specialists[variant] = model.to(device).eval()

    # The paired dataset validates specialist labels 0..2. Full ground truth
    # remains in `samples`; dummy dataset labels never influence routing.
    dataset = PairedStageDataset([(path, 0) for path, _ in samples], data_root, image_size, transform=None)
    loader = DataLoader(
        dataset, batch_size=batch, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    rows, offset = [], 0
    for rgb, tissue, _ in loader:
        rgb, tissue = rgb.to(device), tissue.to(device)
        gate_logits = gate(rgb)
        if gate_logits.ndim != 2 or gate_logits.shape[1] != 3 or not torch.isfinite(gate_logits).all():
            raise ValueError("The four-class gate must emit three CORN logits")
        baseline = predict_labels(gate_logits, "corn")
        routed = baseline != 0
        final = {}
        for variant, model in specialists.items():
            predicted = torch.zeros_like(baseline)
            if routed.any():
                logits = model(rgb[routed], tissue[routed])
                if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
                    raise ValueError("The Stage-2/3/4 specialist must emit two CORN logits")
                predicted[routed] = predict_labels(logits, "corn") + 1
            final[variant] = predicted.cpu().tolist()
        baseline = baseline.cpu().tolist()
        for i, baseline_index in enumerate(baseline):
            rel_path, true_index = samples[offset + i]
            rows.append({
                "path": rel_path, "fold": fold_idx,
                "true_index": int(true_index), "true_class": CLASS_ORDER[true_index],
                "baseline_pred": int(baseline_index),
                "routed_to_specialist": int(baseline_index != 0),
                "cascade_rgb_pred": int(final["rgb"][i]),
                "cascade_rgb_tissue_pred": int(final["rgb_tissue"][i]),
            })
        offset += len(baseline)
    if offset != len(samples):
        raise RuntimeError("Inference did not cover every test image")
    return rows


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    protocol, config = load_training_protocol(output_dir)
    folds = build_folds(args.data_root, config["num_folds"], config["val_fraction"], config["seed"])
    verify_training_inputs(output_dir, args.data_root, folds, protocol)
    gate_protocol = load_gate_protocol(args.data_root, config, folds)
    checkpoint_paths = {}
    for fold_idx in args.folds:
        paths = {"gate": args.gate_checkpoints_dir / f"mit_b1__corn_fold{fold_idx}.pt"}
        paths.update({
            variant: output_dir / "checkpoints" / f"mit_b1__corn__stage234__{variant}_fold{fold_idx}.pt"
            for variant in VARIANTS
        })
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} checkpoint for fold {fold_idx}: {path}")
        if sha256_file(paths["gate"]) != gate_protocol["folds"][str(fold_idx)]["baseline_sha256"]:
            raise ValueError(f"Gate checkpoint for fold {fold_idx} differs from the verified original baseline")
        protocol_fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        for variant in VARIANTS:
            completion_path = output_dir / "results" / "folds" / f"{variant}_fold{fold_idx}.json"
            if not completion_path.is_file():
                raise FileNotFoundError(f"Specialist training is incomplete: {completion_path}")
            completion = json.loads(completion_path.read_text())
            if completion.get("protocol_fingerprint") != protocol_fingerprint:
                raise ValueError(f"Specialist {variant} fold {fold_idx} belongs to a different training protocol")
            if sha256_file(paths[variant]) != completion.get("checkpoint_sha256"):
                raise ValueError(f"Specialist {variant} fold {fold_idx} checkpoint differs from the completed training run")
        checkpoint_paths[fold_idx] = paths
    device = get_device()
    predictions = []
    for fold_idx in args.folds:
        samples = folds[fold_idx]["test"]
        print(f"[cascade fold{fold_idx}] {len(samples)} held-out images, device={device}", flush=True)
        predictions.extend(evaluate_fold(
            fold_idx, samples, args.data_root, config["image_size"], checkpoint_paths[fold_idx],
            device, args.batch, args.workers,
        ))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics, per_class, confusion = build_reports(predictions)
    result_dir = output_dir / "results"
    for name, rows in (
        ("predictions", predictions), ("metrics", metrics),
        ("per_class_metrics", per_class), ("confusion_matrices", confusion),
    ):
        write_csv(result_dir / f"cascade_{name}.csv", rows)
    audit = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "routing": "gate decoded index 0 => Stage 1; otherwise specialist decoded index + 1",
        "gate": "existing fixed four-class RGB MiT-B1 + CORN; no new gate training",
        "classes": CLASS_ORDER, "specialist_classes": config["classes"],
        "selected_folds": args.folds, "n_test": len(predictions),
        "image_size": config["image_size"], "data_root": str(args.data_root.resolve()),
        "training_protocol_sha256": sha256_file(result_dir / "run_protocol.json"),
        "dataset_fingerprint": protocol["dataset_fingerprint"],
        "split_fingerprint": protocol["split_fingerprint"],
        "checkpoint_sha256": {
            str(fold_idx): {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                            for name, path in paths.items()}
            for fold_idx, paths in checkpoint_paths.items()
        },
        "stage1_gate_predictions_preserved": all(
            row["baseline_pred"] != 0 or row["cascade_rgb_pred"] == row["cascade_rgb_tissue_pred"] == 0
            for row in predictions
        ),
        "n_routed_to_specialist": sum(row["routed_to_specialist"] for row in predictions),
        "true_stage1_misrouted": sum(row["true_index"] == 0 and row["baseline_pred"] != 0 for row in predictions),
        "true_stage234_blocked_by_gate": sum(row["true_index"] != 0 and row["baseline_pred"] == 0 for row in predictions),
        "scope": "Full four-stage test classification, including every gate error. Four-class macro-F1.",
    }
    (result_dir / "cascade_protocol.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote cascade reports to {result_dir}", flush=True)
    for row in metrics:
        if row["fold"] == "pooled":
            print(f"{row['model']}: accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
