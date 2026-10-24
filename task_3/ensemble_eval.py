#!/usr/bin/env python3
"""Zero-retraining ensemble evaluation for the wound stage classification grid.

Loads already-trained checkpoints for a chosen set of (encoder, head) configs,
runs each on every fold's held-out test split, converts each model's raw
output into a full 4-class probability vector (softmax head: softmax;
CORN head: decoded from its cumulative conditional probabilities), averages
those vectors across models (sum-rule ensembling, Kittler et al. 1998), and
reports the ensemble's accuracy/macro-F1/MAE/QWK next to each individual
model's own numbers on the exact same held-out samples.

Only configs whose checkpoint exists for every requested fold are usable --
run with no --configs to auto-pick whichever config_ids have all `num_folds`
checkpoints present under configs/base.yaml's checkpoint_dir.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR))

from src.dataset import StageDataset  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.model import build_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402


def corn_logits_to_class_probs(logits, num_classes):
    """logits: (B, num_classes-1) CORN head output.

    sigmoid(logits)[:, k] is the conditional probability P(y > k | y > k-1).
    cumprod gives the unconditional survival probabilities S_k = P(y > k).
    Class probabilities fall out of consecutive differences of S (telescopes
    to exactly 1 by construction, matching coral_pytorch's own
    corn_label_from_logits decoding).
    """
    cond = torch.sigmoid(logits)
    survival = torch.cumprod(cond, dim=1)  # (B, num_classes-1): S_0..S_{K-2}
    ones = torch.ones(survival.shape[0], 1, device=logits.device, dtype=survival.dtype)
    zeros = torch.zeros(survival.shape[0], 1, device=logits.device, dtype=survival.dtype)
    s_prev = torch.cat([ones, survival], dim=1)  # S_{-1}=1, S_0..S_{K-2}
    s_next = torch.cat([survival, zeros], dim=1)  # S_0..S_{K-2}, S_{K-1}=0
    probs = (s_prev - s_next).clamp(min=0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)


def logits_to_class_probs(logits, head, num_classes):
    if head == "softmax":
        return F.softmax(logits, dim=1)
    if head == "corn":
        return corn_logits_to_class_probs(logits, num_classes)
    raise ValueError(f"unknown head {head!r}")


def find_complete_configs(grid, checkpoint_dir, num_folds):
    complete = []
    for cfg in grid:
        if all((checkpoint_dir / f"{cfg['config_id']}_fold{i}.pt").exists() for i in range(num_folds)):
            complete.append(cfg["config_id"])
    return complete


@torch.no_grad()
def collect_probs(model, loader, device, head, num_classes):
    model.eval()
    all_probs, all_true = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = logits_to_class_probs(logits, head, num_classes)
        all_probs.append(probs.cpu().numpy())
        all_true.extend(targets.tolist())
    return np.concatenate(all_probs, axis=0), np.asarray(all_true)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--configs", default=None,
                    help="comma-separated config_id list to ensemble, e.g. "
                         "'mit_b1__corn,mit_b1__softmax' -- default: auto-pick every "
                         "config with all folds' checkpoints present locally")
    p.add_argument("--model-weights", default=None,
                    help="comma-separated per-model ensemble weights, same order/length as "
                         "--configs (default: uniform average)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-csv", default=None,
                    help="optional path to append the ensemble's per-fold rows as CSV")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(TASK_DIR / "configs" / "base.yaml")
    resolved_path = TASK_DIR / "configs" / "resolved_grid.json"
    grid = json.loads(resolved_path.read_text())
    by_id = {c["config_id"]: c for c in grid}

    checkpoint_dir = TASK_DIR / base_cfg["checkpoint_dir"]
    if not checkpoint_dir.exists():
        checkpoint_dir = TASK_DIR / "results" / "task_3_outputs" / "checkpoints"

    num_folds = base_cfg["num_folds"]
    num_classes = len(base_cfg["classes"])

    if args.configs is not None:
        config_ids = [c.strip() for c in args.configs.split(",") if c.strip()]
        missing = [c for c in config_ids if c not in by_id]
        if missing:
            raise SystemExit(f"unknown config_id(s): {missing}")
        incomplete = [c for c in config_ids
                      if not all((checkpoint_dir / f"{c}_fold{i}.pt").exists() for i in range(num_folds))]
        if incomplete:
            raise SystemExit(f"missing checkpoint(s) for: {incomplete} (need all {num_folds} folds)")
    else:
        config_ids = find_complete_configs(grid, checkpoint_dir, num_folds)
        if len(config_ids) < 2:
            raise SystemExit(f"fewer than 2 configs have complete local checkpoints: {config_ids}")
        print(f"auto-picked ensemble members (all {num_folds} folds present locally): {config_ids}")

    if args.model_weights is not None:
        model_weights = [float(w) for w in args.model_weights.split(",")]
        if len(model_weights) != len(config_ids):
            raise SystemExit("--model-weights must have the same length as --configs")
    else:
        model_weights = [1.0] * len(config_ids)
    model_weights = np.asarray(model_weights) / sum(model_weights)

    repo_root = TASK_DIR.parent
    data_root = Path(args.data_root) if args.data_root else repo_root / "datasets" / "wound-stage-classification"

    device = get_device()
    print(f"device={device} ensemble={list(zip(config_ids, model_weights.tolist()))}")

    folds = build_folds(data_root, num_folds, base_cfg["val_fraction"], base_cfg["seed"])

    per_fold_ensemble, per_model_per_fold = [], {cid: [] for cid in config_ids}

    for fold_idx, fold in enumerate(folds):
        test_ds = StageDataset(fold["test"], data_root, base_cfg["image_size"], transform=None)
        test_loader = DataLoader(test_ds, batch_size=base_cfg["batch_size"], shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

        weighted_probs, y_true = None, None
        for cid, w in zip(config_ids, model_weights):
            cfg = by_id[cid]
            model = build_model(cfg["encoder"], None, cfg["head"], num_classes=num_classes)
            state = torch.load(checkpoint_dir / f"{cid}_fold{fold_idx}.pt", map_location=device)
            model.load_state_dict(state)
            model.to(device)

            probs, y_true = collect_probs(model, test_loader, device, cfg["head"], num_classes)
            weighted_probs = probs * w if weighted_probs is None else weighted_probs + probs * w

            model_preds = probs.argmax(axis=1)
            model_metrics = compute_metrics(y_true, model_preds, num_classes)
            per_model_per_fold[cid].append(model_metrics)
            print(f"  [fold{fold_idx}] {cid:28s} acc={model_metrics['accuracy']:.4f} "
                  f"f1={model_metrics['macro_f1']:.4f} qwk={model_metrics['qwk']:.4f}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        ensemble_preds = weighted_probs.argmax(axis=1)
        ensemble_metrics = compute_metrics(y_true, ensemble_preds, num_classes)
        per_fold_ensemble.append(ensemble_metrics)
        print(f"  [fold{fold_idx}] {'ENSEMBLE':28s} acc={ensemble_metrics['accuracy']:.4f} "
              f"f1={ensemble_metrics['macro_f1']:.4f} qwk={ensemble_metrics['qwk']:.4f}")

    def summarize(label, metrics_list):
        acc = np.array([m["accuracy"] for m in metrics_list])
        f1 = np.array([m["macro_f1"] for m in metrics_list])
        mae = np.array([m["mae"] for m in metrics_list])
        qwk = np.array([m["qwk"] for m in metrics_list])
        print(f"{label:28s} acc={acc.mean():.4f}±{acc.std():.4f}  f1={f1.mean():.4f}±{f1.std():.4f}  "
              f"mae={mae.mean():.4f}±{mae.std():.4f}  qwk={qwk.mean():.4f}±{qwk.std():.4f}")

    print("\n=== 3-fold summary ===")
    for cid in config_ids:
        summarize(cid, per_model_per_fold[cid])
    summarize("ENSEMBLE", per_fold_ensemble)

    if args.out_csv:
        import csv
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_path.exists()
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["ensemble_members", "weights", "fold", "accuracy", "macro_f1", "mae", "qwk"])
            for fold_idx, m in enumerate(per_fold_ensemble):
                writer.writerow(["+".join(config_ids), "+".join(f"{w:.3f}" for w in model_weights),
                                  fold_idx, round(m["accuracy"], 4), round(m["macro_f1"], 4),
                                  round(m["mae"], 4), round(m["qwk"], 4)])


if __name__ == "__main__":
    main()
