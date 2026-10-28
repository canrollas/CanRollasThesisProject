#!/usr/bin/env python3
"""Validation-only training. Test folds are accessed by evaluate.py separately."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

SCRIPTS = Path(__file__).resolve().parent
EXP_DIR = SCRIPTS.parent
REPO_ROOT = SCRIPTS.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))
from src.losses import build_loss
from src.metrics import compute_metrics, predict_labels
from src.splits import build_folds
from src.utils import get_device, load_yaml, set_seed

from data import StagePriorDataset
from model import TissueGeometryTokenModel


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_digest(fold):
    return hashlib.sha256(json.dumps(fold, sort_keys=True).encode()).hexdigest()


def input_fingerprint(cfg, samples, mode):
    """Bind the exact source/cache bytes, independent of absolute location."""
    kinds = [("rgb", "data_root", False), ("tissue", "tissue_root", True)]
    if mode in ("full", "geometry"):
        kinds.append(("depth", "depth_root", True))
    result = {}
    for label, root_key, npy in kinds:
        digest = hashlib.sha256()
        for relative, _ in sorted(samples):
            rel = Path(relative).with_suffix(".npy") if npy else Path(relative)
            content = sha256(Path(cfg[root_key]) / rel)
            digest.update(f"{rel.as_posix()}:{content}\n".encode())
        result[label] = digest.hexdigest()
    for label, root_key, _ in kinds:
        for filename in ("provenance.json", "manifest.json"):
            path = Path(cfg[root_key]) / filename
            if path.is_file():
                result[f"{label}/{filename}"] = sha256(path)
    return result


def code_fingerprint():
    return {name: sha256(SCRIPTS / name) for name in ("train.py", "model.py", "data.py")}


def validate_baseline_protocol(cfg, folds):
    """Historical checkpoints cannot be reused under a different outer split."""
    expected = json.loads((EXP_DIR / "configs/baseline_protocol.json").read_text())
    for key in ("split_seed", "num_folds", "val_fraction", "image_size"):
        if cfg[key] != expected[key]:
            raise ValueError(f"Historical baseline requires {key}={expected[key]}")
    for idx, fold in enumerate(folds):
        if split_digest(fold) != expected["folds"][str(idx)]["split_sha256"]:
            raise ValueError(f"Fold{idx} membership differs from the historical baseline")
        path = Path(cfg["rgb_checkpoint_template"].format(fold=idx))
        if sha256(path) != expected["folds"][str(idx)]["baseline_sha256"]:
            raise ValueError(f"Fold{idx} RGB checkpoint differs from verified historical weights")


def resolve_config(path):
    cfg = load_yaml(path)
    for name in ("data_root", "tissue_root", "depth_root", "rgb_checkpoint_template"):
        cfg[name] = str((EXP_DIR / cfg[name]).resolve())
    return cfg


def model_from_config(cfg, mode):
    return TissueGeometryTokenModel(
        mode=mode, token_dim=cfg["token_dim"], dropout=cfg["dropout"],
        residual_scale=cfg["residual_scale"],
    )


def make_loader(samples, cfg, mode, train, workers=0, batch_size=None):
    ds = StagePriorDataset(
        samples, cfg["data_root"], cfg["tissue_root"],
        cfg["depth_root"] if mode in ("full", "geometry") else None,
        image_size=cfg["image_size"], augment=train, require_depth=(mode in ("full", "geometry")),
    )
    return DataLoader(ds, batch_size=batch_size or cfg["batch_size"],
                      shuffle=train, num_workers=workers, pin_memory=torch.cuda.is_available())


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def forward_batch(model, batch):
    return model(batch["rgb"], batch["tissue"], batch["depth"], batch["depth_valid"])


def run_epoch(model, loader, device, cfg, optimizer=None, baseline_only=False):
    model.train(optimizer is not None)
    criterion = build_loss("corn", 4)
    losses, targets, predictions, records = [], [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for raw in loader:
            batch = move_batch(raw, device)
            out = forward_batch(model, batch)
            logits = out["rgb_logits"] if baseline_only else out["logits"]
            loss = criterion(logits, batch["target"])
            active = out["has_wound"].reshape(-1).bool()
            if optimizer is not None and model.mode != "rgb" and active.any():
                loss = loss + cfg["aux_weight"] * criterion(
                    out["local_logits"][active], batch["target"][active])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; refusing to save a result")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
            pred = predict_labels(logits.detach(), "corn")
            targets.extend(batch["target"].cpu().tolist())
            predictions.extend(pred.cpu().tolist())
            losses.append((float(loss.detach()), len(pred)))
            for name, target, prediction in zip(batch["path"], batch["target"].cpu().tolist(), pred.cpu().tolist()):
                records.append({"path": name, "target": target, "prediction": prediction})
    stats = compute_metrics(targets, predictions, 4)
    stats["loss"] = sum(loss * count for loss, count in losses) / max(sum(c for _, c in losses), 1)
    stats["confusion"] = stats["confusion"].tolist()
    return stats, records


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=EXP_DIR / "configs/base.yaml")
    p.add_argument("--mode", choices=("rgb", "tissue", "geometry", "full"), default="full")
    p.add_argument("--folds", default="0,1,2")
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--train-seed", type=int)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--tissue-root", type=Path)
    p.add_argument("--depth-root", type=Path)
    p.add_argument("--output-dir", type=Path, default=EXP_DIR / "runs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="8 train + 8 validation samples; separate smoke output")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config(args.config)
    for name in ("data_root", "tissue_root", "depth_root"):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = str(value.resolve())
    for name in ("epochs", "batch_size", "train_seed"):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("epochs and batch-size must be positive")
    selected = [int(f) for f in args.folds.split(",")]
    if len(selected) != len(set(selected)) or any(f not in range(3) for f in selected):
        raise ValueError("folds must be unique indices from 0,1,2")
    if args.smoke:
        cfg["epochs"] = 1
        cfg["batch_size"] = 4
        selected = selected[:1]
    folds = build_folds(cfg["data_root"], cfg["num_folds"], cfg["val_fraction"], cfg["split_seed"])
    validate_baseline_protocol(cfg, folds)
    all_samples = [sample for fold in folds for sample in fold["test"]]
    data_identity = input_fingerprint(cfg, all_samples, args.mode)
    device = torch.device(args.device) if args.device else get_device()
    output = args.output_dir.resolve() / ("smoke" if args.smoke else "main") / f"seed{cfg['train_seed']}" / args.mode
    output.mkdir(parents=True, exist_ok=True)
    print(f"mode={args.mode} device={device} folds={selected} output={output}", flush=True)
    summary = []
    for fold_idx in selected:
        set_seed(cfg["train_seed"] + fold_idx)
        fold = folds[fold_idx]
        # Smoke membership is deterministic, always disjoint, never reaches test.
        train_samples = fold["train"][:8] if args.smoke else fold["train"]
        val_samples = fold["val"][:8] if args.smoke else fold["val"]
        baseline_path = Path(cfg["rgb_checkpoint_template"].format(fold=fold_idx))
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Fold-specific RGB checkpoint required: {baseline_path}")
        train_loader = make_loader(train_samples, cfg, args.mode, True, args.workers)
        val_loader = make_loader(val_samples, cfg, args.mode, False, args.workers)
        # Separate validation pass for initialization: original RGB pathway only.
        model = model_from_config(cfg, args.mode)
        model.load_rgb_baseline(baseline_path)
        model.set_last_stage_trainable(True)
        model.to(device)
        rgb_params, prior_params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                (rgb_params if name.startswith(("rgb_backbone.", "rgb_head.")) else prior_params).append(param)
        groups = [{"params": rgb_params, "lr": cfg["backbone_lr"]}]
        if prior_params and args.mode != "rgb":
            groups.append({"params": prior_params, "lr": cfg["head_lr"]})
        optimizer = AdamW(groups, weight_decay=cfg["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, cfg["epochs"], eta_min=cfg["backbone_lr"] * cfg["eta_min_factor"])
        best_path = output / f"fold{fold_idx}.pt"
        resume_path = output / f"fold{fold_idx}.resume.pt"
        history_path = output / f"fold{fold_idx}.history.json"
        fingerprint = {"config": cfg, "mode": args.mode, "fold": fold_idx,
                       "split_sha256": split_digest(fold), "baseline_sha256": sha256(baseline_path),
                       "smoke": args.smoke, "inputs": data_identity, "code": code_fingerprint()}
        start, best_score, best_epoch, bad_epochs, history = 1, -math.inf, 0, 0, []
        if best_path.exists() and not args.resume:
            raise FileExistsError(f"Run exists: {best_path}. Use --resume or a different --output-dir")
        if args.resume:
            if not resume_path.exists():
                raise FileNotFoundError(f"No resumable run: {resume_path}")
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            if saved["fingerprint"] != fingerprint:
                raise ValueError("Resume configuration, data split or baseline differs")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            start, best_score, best_epoch = saved["epoch"] + 1, saved["best_score"], saved["best_epoch"]
            bad_epochs, history = saved["bad_epochs"], saved["history"]
            import random
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"])
            if device.type == "cuda" and saved.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            if device.type == "mps" and saved.get("mps_rng") is not None:
                torch.mps.set_rng_state(saved["mps_rng"])
            if bad_epochs >= cfg["patience"]:
                start = cfg["epochs"] + 1
                print(f"fold{fold_idx} already early-stopped; keeping selected checkpoint", flush=True)
        else:
            baseline_stats, _ = run_epoch(model, val_loader, device, cfg, baseline_only=True)
            print(f"fold{fold_idx} original RGB validation={baseline_stats}", flush=True)
        t0 = time.time()
        for epoch in range(start, cfg["epochs"] + 1):
            train_stats, _ = run_epoch(model, train_loader, device, cfg, optimizer)
            val_stats, records = run_epoch(model, val_loader, device, cfg)
            scheduler.step()
            score = val_stats["macro_f1"]
            if score > best_score:
                best_score, best_epoch, bad_epochs = score, epoch, 0
                torch.save({"model": model.state_dict(), "fingerprint": fingerprint,
                            "best_epoch": epoch, "validation": val_stats}, best_path.with_suffix(".tmp"))
                best_path.with_suffix(".tmp").replace(best_path)
                (output / f"fold{fold_idx}.validation_predictions.json").write_text(json.dumps(records, indent=2))
            else:
                bad_epochs += 1
            history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
            history_path.write_text(json.dumps(history, indent=2))
            import random
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "fingerprint": fingerprint,
                     "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch,
                     "bad_epochs": bad_epochs, "history": history,
                     "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None}
            torch.save(state, resume_path.with_suffix(".tmp"))
            resume_path.with_suffix(".tmp").replace(resume_path)
            print(f"fold{fold_idx} epoch={epoch} train_loss={train_stats['loss']:.4f} "
                  f"val_acc={val_stats['accuracy']:.4f} val_f1={score:.4f} best={best_score:.4f}", flush=True)
            if bad_epochs >= cfg["patience"]:
                break
        summary.append({"fold": fold_idx, "best_epoch": best_epoch, "best_val_f1": best_score,
                        "seconds_this_invocation": time.time() - t0, "test_evaluated": False})
        (output / f"fold{fold_idx}.summary.json").write_text(json.dumps(summary[-1], indent=2))
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
