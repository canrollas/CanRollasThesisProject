#!/usr/bin/env python3
"""Retrain matched Stage-2/3/4 RGB and RGB+tissue models (six full runs)."""
import argparse
import gc
import hashlib
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))
sys.path.insert(0, str(THIS_DIR))

from src.augmentations import ImageAugmentation  # noqa: E402
from src.utils import get_device, load_yaml, set_seed  # noqa: E402
from data import (  # noqa: E402
    CLASSES, PairedStageDataset, balanced_subset, build_stage234_folds,
    inspect_dataset, make_manifest, seed_worker,
)
from engine import fit, run_epoch  # noqa: E402
from model import VARIANTS, build_variant  # noqa: E402
from reports import write_reports  # noqa: E402


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checkpoint_digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_loaders(fold, data_root, config, workers, device, fold_seed):
    loaders = []
    for offset, split in enumerate(("train", "val", "test")):
        transform = ImageAugmentation(**config["augment"]) if split == "train" else None
        ds = PairedStageDataset(fold[split], data_root, config["image_size"], transform=transform)
        generator = torch.Generator().manual_seed(fold_seed + offset * 1000)
        loaders.append(DataLoader(
            ds, batch_size=config["batch_size"], shuffle=split == "train", num_workers=workers,
            pin_memory=device.type == "cuda", persistent_workers=False,
            worker_init_fn=seed_worker, generator=generator,
        ))
    return loaders


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="Exp 3 uint8 eight-channel .npy dataset; keep ALL four stage folders")
    parser.add_argument("--output-dir", default=str(EXP_DIR), help="Drive-backed output directory in Colab")
    parser.add_argument("--variants", default="rgb,rgb_tissue")
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--weights", choices=("imagenet", "none"), default=None, help="Use none only for offline smoke checks")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true", help="Tiny balanced first fold, default 2 epochs; isolated output-dir/smoke")
    args = parser.parse_args()
    args.variants = list(dict.fromkeys(args.variants.split(",")))
    if not args.variants or any(v not in VARIANTS for v in args.variants):
        parser.error("--variants must contain rgb and/or rgb_tissue")
    try:
        args.folds = list(dict.fromkeys(int(f) for f in args.folds.split(",")))
    except ValueError:
        parser.error("--folds must be comma-separated integers")
    if not args.folds or any(f not in range(3) for f in args.folds):
        parser.error("--folds must be selected from 0,1,2")
    if args.workers < 0 or any(v is not None and v <= 0 for v in (args.batch, args.epochs)):
        parser.error("Batch/epochs must be positive and workers non-negative")
    return args


def main():
    args = parse_args()
    config = load_yaml(EXP_DIR / "configs" / "base.yaml")
    if config["classes"] != CLASSES or config["head"] != "corn":
        raise ValueError("This experiment requires Stage 2/3/4 and a three-class CORN head")
    if args.batch is not None:
        config["batch_size"] = args.batch
    config["epochs"] = args.epochs if args.epochs is not None else (2 if args.smoke else config["epochs"])
    if args.weights is not None:
        config["weights"] = None if args.weights == "none" else args.weights
    config["workers"] = args.workers
    data_root = (Path(args.data_root) if args.data_root else EXP_DIR / config.pop("data_root")).resolve()
    config.pop("data_root", None)
    output_dir = Path(args.output_dir).resolve()
    if args.smoke:
        output_dir = output_dir / "smoke"
        args.folds = args.folds[:1]
    results_dir, checkpoints = output_dir / "results", output_dir / "checkpoints"
    (results_dir / "folds").mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else get_device()

    source = inspect_dataset(data_root)
    folds = build_stage234_folds(data_root, config["num_folds"], config["val_fraction"], config["seed"])
    if args.smoke:
        folds = [{split: balanced_subset(samples, 3 if split == "train" else 2)
                  for split, samples in fold.items()} for fold in folds]
    manifest = make_manifest(folds)
    protocol = {
        "experiment": "exp15_stage234_tissue", "protocol_version": 1,
        "config": config, "num_classes": 3, "corn_logits": 2, "smoke": args.smoke,
        "dataset_fingerprint": source["fingerprint"], "dataset_class_counts": source["class_counts"],
        "split_fingerprint": digest(manifest), "source_provenance": source["source_provenance"],
        "checkpoint_selection": "maximum validation macro-F1 over three classes; first epoch wins ties",
        "variants": list(VARIANTS), "fold_training_seed": "seed + fold_index",
        "rgb_input": "same unmasked channels 0..2 from Exp 3 .npy for both variants",
        "fusion": "Exp 4 concat: RGB backbone + five-channel TissueCNN + MLP + two-logit CORN",
    }
    protocol_path = results_dir / "run_protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if previous != protocol:
            raise ValueError("Output directory already contains a different protocol/data/split. "
                             "Use a new --output-dir; existing results and checkpoints have been preserved.")
    else:
        atomic_json(protocol_path, protocol)
    atomic_json(results_dir / "split_manifest.json", manifest)
    fingerprint = digest(protocol)
    atomic_json(results_dir / "runtime.json", {
        "last_invocation_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": str(torch.__version__), "device": str(device),
        "cuda": torch.version.cuda, "data_root": str(data_root), "variants_requested": args.variants,
        "folds_requested": args.folds,
    })
    print(f"device={device} classes={CLASSES} epochs={config['epochs']} batch={config['batch_size']}", flush=True)
    print(f"data_counts={source['class_counts']} output={output_dir}", flush=True)
    if source["source_provenance"] is None:
        print("NOTE: Dataset provenance.json missing; source provenance cannot be verified automatically.", flush=True)
    for idx in args.folds:
        print(f"fold{idx}: " + " ".join(f"{split}={len(samples)}" for split, samples in folds[idx].items()), flush=True)

    failures = []
    for fold_idx in args.folds:
        for variant in args.variants:
            config_id = f"mit_b1__corn__stage234__{variant}"
            bundle_path = results_dir / "folds" / f"{variant}_fold{fold_idx}.json"
            checkpoint_path = checkpoints / f"{config_id}_fold{fold_idx}.pt"
            resume_path = checkpoints / f"{config_id}_fold{fold_idx}.resume.pt"
            if bundle_path.exists():
                bundle = json.loads(bundle_path.read_text())
                if bundle["protocol_fingerprint"] != fingerprint:
                    raise ValueError(f"Incompatible completed fold: {bundle_path}")
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Completed fold's checkpoint is missing: {checkpoint_path}")
                print(f"[{variant} fold{fold_idx}] already complete, skipping", flush=True)
                write_reports(output_dir)
                resume_path.unlink(missing_ok=True)
                continue
            model = loaders = None
            try:
                fold_seed = config["seed"] + fold_idx
                set_seed(fold_seed)
                loaders = make_loaders(folds[fold_idx], data_root, config, args.workers, device, fold_seed)
                model = build_variant(variant, weights=config["weights"], encoder=config["encoder"])
                fitted = fit(
                    model, loaders[0], loaders[1], device, config, checkpoint_path, resume_path,
                    fingerprint, results_dir / f"history_{variant}_fold{fold_idx}.csv",
                    log_prefix=f"[{variant} fold{fold_idx}]",
                )
                test = run_epoch(model, loaders[2], device, collect=True)
                predictions = []
                for (path, label), true, pred, probs in zip(
                    folds[fold_idx]["test"], test["y_true"], test["y_pred"], test["probabilities"], strict=True,
                ):
                    if label != true:
                        raise RuntimeError("Test sample order changed")
                    predictions.append({
                        "variant": variant, "fold": fold_idx, "path": path,
                        "true_index": true, "true_stage": CLASSES[true],
                        "pred_index": pred, "pred_stage": CLASSES[pred],
                        **{f"prob_{stage}": p for stage, p in zip(CLASSES, probs, strict=True)},
                    })
                bundle = {
                    "config_id": config_id, "variant": variant, "encoder": config["encoder"],
                    "head": "corn", "weights": config["weights"], "fold": fold_idx,
                    "protocol_fingerprint": fingerprint, "n_test": len(predictions),
                    "checkpoint_sha256": checkpoint_digest(checkpoint_path),
                    "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
                    **{key: test[key] for key in ("accuracy", "macro_f1", "mae", "qwk")},
                    **fitted, "confusion": test["confusion"].tolist(), "predictions": predictions,
                }
                atomic_json(checkpoint_path.with_suffix(".provenance.json"), {
                    "config_id": config_id, "fold": fold_idx, "protocol_fingerprint": fingerprint,
                    "checkpoint_sha256": bundle["checkpoint_sha256"],
                    "trained_at_utc": datetime.now(timezone.utc).isoformat(), "config": config, **fitted,
                })
                atomic_json(bundle_path, bundle)
                write_reports(output_dir)
                resume_path.unlink(missing_ok=True)
                print(f"[{variant} fold{fold_idx}] DONE test_f1={test['macro_f1']:.4f} "
                      f"test_acc={test['accuracy']:.4f}", flush=True)
            except Exception as exc:
                traceback.print_exc()
                failure = {"variant": variant, "fold": fold_idx, "error": f"{type(exc).__name__}: {exc}",
                           "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                failures.append(failure)
                with (results_dir / "failures.jsonl").open("a") as handle:
                    handle.write(json.dumps(failure) + "\n")
            finally:
                del model, loaders
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    write_reports(output_dir)
    if failures:
        raise SystemExit(f"{len(failures)} run(s) failed. Read results/failures.jsonl and rerun the same command to resume.")
    print(f"All requested runs complete. Reports: {results_dir}", flush=True)


if __name__ == "__main__":
    main()
