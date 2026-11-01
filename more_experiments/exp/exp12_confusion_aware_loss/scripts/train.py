#!/usr/bin/env python3
"""Experiment 12 (Chapter 8): Confusion-Aware CORN Loss.

Trains the EXACT SAME mit_b1 + CORN architecture, data root, image size,
augmentation protocol, and 3-fold split (seed=42) as Chapter 7's baseline
(task_3/run_ablation.py's mit_b1__corn config) -- see configs/base.yaml,
copied from task_3/configs/base.yaml verbatim. The only thing this script
changes between runs is the training LOSS (scripts/losses.py's
CombinedLoss): 'baseline' reproduces the original CornLoss exactly,
'A'/'B'/'C' add one auxiliary term (ordinal-distance / confusion-aware /
boundary-weighted respectively -- see losses.py's own docstring).

Reuses task_3/src's model/dataset/augmentations/splits/metrics/utils
UNCHANGED. engine.py is a LOCAL copy of task_3/src/engine.py's fit()/
run_epoch() -- the only difference is that the criterion is passed in by
the caller instead of rebuilt internally, since task_3's own fit() cannot
be handed a CombinedLoss (see engine.py's docstring).

Two modes, per the brief's "do not select hyperparameters using test
results":
  --mode select  trains on {train}, evaluates each epoch (and reports the
                  final best-val-macro-F1 checkpoint's full metrics) on
                  {val} ONLY. Never touches {test}. Appends one row to
                  results/selection_results.csv. Use this to run the small
                  lambda/alpha/beta grids in the brief's Section 4 before
                  choosing a configuration.
  --mode final    same training procedure, but the resulting best-val
                  checkpoint is THEN evaluated once on the outer {test}
                  split, the checkpoint is saved, every test-set
                  prediction (path, true/pred label, CORN logits,
                  conditional + class probabilities) is written to
                  results/predictions/, and a row is appended to
                  results/results.csv. Run this ONCE per variant/fold
                  after --mode select has already picked lam/alpha/beta.

Usage examples:
  python test_losses.py                                  # run first
  python build_cost_matrices.py                           # run once, before any variant='B' run
  python train.py --variant baseline --fold 0 --mode final
  python train.py --variant A --lam 0.1 --fold 0 --mode select
  python train.py --variant B --lam 0.1 --alpha 1 --fold 0 --mode select
  python train.py --variant C --beta 0.5 --fold 0 --mode select
  python train.py --variant B --lam 0.1 --alpha 1 --fold 0 --mode final   # after selection

Safe to interrupt and resume (--mode final only; --mode select runs are
cheap enough, and their log rows aren't meant to be a resumable grid-search
ledger), same pattern as the other exp/*/scripts/train.py.
"""
import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent  # exp/exp12_confusion_aware_loss/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))
sys.path.insert(0, str(THIS_DIR))

from src.augmentations import ImageAugmentation  # noqa: E402
from src.dataset import StageDataset  # noqa: E402
from src.metrics import predict_labels  # noqa: E402
from src.model import build_model, count_params_m  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml, set_seed  # noqa: E402

from engine import fit, run_epoch  # noqa: E402
from losses import CombinedLoss, corn_class_probabilities  # noqa: E402

RESULT_FIELDS = [
    "config_id", "variant", "lam", "alpha", "beta", "fold", "params_m",
    "accuracy", "macro_f1", "mae", "qwk", "train_time_s",
]
SELECTION_FIELDS = [
    "config_id", "variant", "lam", "alpha", "beta", "fold",
    "val_accuracy", "val_macro_f1", "val_mae", "val_qwk", "train_time_s",
]


def config_id(variant, lam, alpha, beta):
    if variant == "baseline":
        return "mit_b1__corn__exp12_baseline"
    if variant == "A":
        return f"mit_b1__corn__exp12A_lam{lam:g}"
    if variant == "B":
        return f"mit_b1__corn__exp12B_lam{lam:g}_alpha{alpha:g}"
    if variant == "C":
        return f"mit_b1__corn__exp12C_beta{beta:g}"
    raise ValueError(variant)


def load_cost_matrix(fold_idx, alpha):
    path = EXP_DIR / "results" / "cost_matrices" / f"fold{fold_idx}" / f"cost_matrix_alpha{alpha:g}.json"
    if not path.is_file():
        raise SystemExit(
            f"cost matrix not found: {path}\nRun scripts/build_cost_matrices.py first "
            "(it only needs the existing baseline checkpoints, no training)."
        )
    with open(path) as f:
        data = json.load(f)
    return torch.tensor(data["cost_matrix"], dtype=torch.float32)


def build_criterion(variant, num_classes, fold_idx, lam, alpha, beta):
    if variant == "baseline":
        return CombinedLoss(num_classes, variant="baseline")
    if variant == "A":
        return CombinedLoss(num_classes, variant="A", lam=lam)
    if variant == "B":
        cost_matrix = load_cost_matrix(fold_idx, alpha)
        return CombinedLoss(num_classes, variant="B", lam=lam, cost_matrix=cost_matrix)
    if variant == "C":
        return CombinedLoss(num_classes, variant="C", beta=beta)
    raise ValueError(variant)


def make_loaders(data_root, fold, base_cfg, batch_size, workers, smoke):
    aug = ImageAugmentation(**base_cfg["augment"])
    train_samples, val_samples, test_samples = fold["train"], fold["val"], fold["test"]
    if smoke:
        train_samples, val_samples, test_samples = train_samples[:8], val_samples[:4], test_samples[:4]

    train_ds = StageDataset(train_samples, data_root, base_cfg["image_size"], transform=aug)
    val_ds = StageDataset(val_samples, data_root, base_cfg["image_size"], transform=None)
    test_ds = StageDataset(test_samples, data_root, base_cfg["image_size"], transform=None)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader, test_samples


def predict_and_save(model, loader, samples, device, out_path, variant, fold_idx, lam, alpha, beta):
    """Writes one row per test-set image: path, fold, variant/hyperparams,
    true/predicted stage, raw CORN logits, conditional probabilities
    (q1..q3), class probabilities (p1..p4), correctness -- the full record
    the brief's Section 5 step 8 and Section 7 require, matched by image ID
    for the later paired comparison against the baseline."""
    model.eval()
    rows = []
    idx = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds = predict_labels(logits, "corn").cpu()
            class_probs, conditional = corn_class_probabilities(logits)
            class_probs = class_probs.cpu()
            conditional = conditional.cpu()
            for i in range(images.shape[0]):
                rel_path, true_label = samples[idx]
                pred_label = int(preds[i].item())
                rows.append({
                    "path": rel_path,
                    "fold": fold_idx,
                    "variant": variant,
                    "lam": lam,
                    "alpha": alpha,
                    "beta": beta,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": true_label == pred_label,
                    "logit_0": float(logits[i, 0].item()),
                    "logit_1": float(logits[i, 1].item()),
                    "logit_2": float(logits[i, 2].item()),
                    "conditional_0": float(conditional[i, 0].item()),
                    "conditional_1": float(conditional[i, 1].item()),
                    "conditional_2": float(conditional[i, 2].item()),
                    "prob_0": float(class_probs[i, 0].item()),
                    "prob_1": float(class_probs[i, 1].item()),
                    "prob_2": float(class_probs[i, 2].item()),
                    "prob_3": float(class_probs[i, 3].item()),
                })
                idx += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_row(csv_path, row, fieldnames):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["baseline", "A", "B", "C"])
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--mode", required=True, choices=["select", "final"])
    p.add_argument("--lam", type=float, default=0.0, help="required for variants A/B")
    p.add_argument("--alpha", type=float, default=0.0, help="required for variant B")
    p.add_argument("--beta", type=float, default=0.0, help="required for variant C")
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = Path(args.data_root) if args.data_root else EXP_DIR / base_cfg["data_root"]
    device = get_device()
    num_classes = len(base_cfg["classes"])
    batch_size = args.batch if args.batch is not None else base_cfg["batch_size"]
    epochs = args.epochs if args.epochs is not None else (2 if args.smoke else base_cfg["epochs"])

    set_seed(base_cfg["seed"])
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    fold = folds[args.fold]

    cid = config_id(args.variant, args.lam, args.alpha, args.beta)
    print(f"device={device} config={cid} mode={args.mode} fold={args.fold} epochs={epochs}", flush=True)

    train_loader, val_loader, test_loader, test_samples = make_loaders(
        data_root, fold, base_cfg, batch_size, args.workers, args.smoke
    )

    model = build_model(base_cfg["encoder"], base_cfg["weights"], base_cfg["head"], num_classes=num_classes)
    params_m = count_params_m(model)
    criterion = build_criterion(args.variant, num_classes, args.fold, args.lam, args.alpha, args.beta)

    try:
        t0 = time.time()
        model, best_val_f1 = fit(
            model, train_loader, val_loader, criterion, device,
            epochs=epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
            eta_min_factor=base_cfg["eta_min_factor"],
            head=base_cfg["head"], num_classes=num_classes,
            log_prefix=f"[{cid} fold{args.fold} {args.mode}]",
            checkpoint_path=None, resume_path=None,
        )
        train_time_s = time.time() - t0

        val_stats = run_epoch(model, val_loader, criterion, device, base_cfg["head"], num_classes, optimizer=None)

        if args.mode == "select":
            row = {
                "config_id": cid, "variant": args.variant, "lam": args.lam, "alpha": args.alpha, "beta": args.beta,
                "fold": args.fold,
                "val_accuracy": round(val_stats["accuracy"], 4),
                "val_macro_f1": round(val_stats["macro_f1"], 4),
                "val_mae": round(val_stats["mae"], 4),
                "val_qwk": round(val_stats["qwk"], 4),
                "train_time_s": round(train_time_s, 1),
            }
            append_row(EXP_DIR / "results" / "selection_results.csv", row, SELECTION_FIELDS)
            print(f"[{cid} fold{args.fold}] SELECT DONE val_f1={row['val_macro_f1']:.4f} "
                  f"val_acc={row['val_accuracy']:.4f} val_qwk={row['val_qwk']:.4f}", flush=True)
            return

        # --mode final: evaluate on the outer test split, save checkpoint + predictions
        test_stats = run_epoch(model, test_loader, criterion, device, base_cfg["head"], num_classes, optimizer=None)

        checkpoint_dir = EXP_DIR / base_cfg["checkpoint_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{cid}_fold{args.fold}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        provenance_path = checkpoint_path.with_suffix(".provenance.json")
        with open(provenance_path, "w") as f:
            json.dump({
                "config_id": cid, "variant": args.variant, "lam": args.lam, "alpha": args.alpha, "beta": args.beta,
                "fold": args.fold,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "encoder": base_cfg["encoder"], "head": base_cfg["head"], "weights": base_cfg["weights"],
                "image_size": base_cfg["image_size"], "batch_size": batch_size, "epochs": epochs,
                "lr": base_cfg["lr"], "weight_decay": base_cfg["weight_decay"],
                "eta_min_factor": base_cfg["eta_min_factor"], "seed": base_cfg["seed"],
                "data_root": str(data_root), "params_m": round(params_m, 3),
                "best_val_macro_f1": round(best_val_f1, 4),
            }, f, indent=2)

        predictions_path = EXP_DIR / "results" / "predictions" / f"{cid}_fold{args.fold}.csv"
        predict_and_save(model, test_loader, test_samples, device, predictions_path,
                          args.variant, args.fold, args.lam, args.alpha, args.beta)

        row = {
            "config_id": cid, "variant": args.variant, "lam": args.lam, "alpha": args.alpha, "beta": args.beta,
            "fold": args.fold, "params_m": round(params_m, 3),
            "accuracy": round(test_stats["accuracy"], 4),
            "macro_f1": round(test_stats["macro_f1"], 4),
            "mae": round(test_stats["mae"], 4),
            "qwk": round(test_stats["qwk"], 4),
            "train_time_s": round(train_time_s, 1),
        }
        append_row(EXP_DIR / "results" / "results.csv", row, RESULT_FIELDS)
        print(f"[{cid} fold{args.fold}] FINAL DONE test_acc={row['accuracy']:.4f} "
              f"test_f1={row['macro_f1']:.4f} test_qwk={row['qwk']:.4f}", flush=True)

    except Exception as exc:  # noqa: BLE001 - visible failure, not a silent crash of a larger grid
        traceback.print_exc()
        print(f"[{cid} fold{args.fold} {args.mode}] FAILED: {exc}", flush=True)
        raise
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
