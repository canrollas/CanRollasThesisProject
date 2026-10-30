#!/usr/bin/env python3
"""Evaluates an already-trained exp12 checkpoint on its fold's held-out
TEST split and (optionally) appends the result as a real row to this
experiment's own results CSV, in the same RESULT_FIELDS format
compare_results.py expects.

Accepts either checkpoint shape:
  - a plain state_dict (what scripts/train.py's final `checkpoint_path`
    saves, via task_3's own `fit()`)
  - a resume dict (`{"epoch", "model_state", ..., "best_macro_f1",
    "best_state"}`, what `fit()` writes after every epoch to
    `resume_path`) -- uses `best_state` (best validation macro_f1 seen so
    far), NOT `model_state` (which is just whatever epoch it stopped at).

Useful when training ran somewhere else (another machine, interrupted
Colab session, etc.) and only the checkpoint file was brought back here --
no re-training needed, this only runs the frozen model forward on the
test split.
"""
import argparse
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.dataset import StageDataset  # noqa: E402
from src.engine import run_epoch  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.model import Backbone, CornHead  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import append_result, get_device, load_yaml, set_seed  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

CONFIG_ID = "mit_b2_task2init__corn"
ENCODER_NAME = "mit_b2"


def load_state(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "best_state" in ckpt:
        source = "best_state"
        epoch = ckpt.get("epoch")
        best_macro_f1 = ckpt.get("best_macro_f1")
        print(f"resume-style checkpoint: stopped after epoch {epoch}, "
              f"best val macro_f1 so far = {best_macro_f1}", flush=True)
        return ckpt["best_state"], source, epoch, best_macro_f1
    return ckpt, "plain_state_dict", None, None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to a .pt or .resume.pt file")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--append", action="store_true", help="append the result row to results/exp12_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    recipe_cfg = load_yaml(REPO_ROOT / "task_3" / "configs" / "base.yaml")
    data_root = Path(args.data_root) if args.data_root else REPO_ROOT / "datasets" / "wound-stage-classification"
    num_classes = len(recipe_cfg["classes"])
    batch_size = args.batch if args.batch is not None else recipe_cfg["batch_size"]

    set_seed(recipe_cfg["seed"])
    folds = build_folds(data_root, recipe_cfg["num_folds"], recipe_cfg["val_fraction"], recipe_cfg["seed"])
    fold = folds[args.fold]

    test_ds = StageDataset(fold["test"], data_root, recipe_cfg["image_size"], transform=None)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=args.workers)
    print(f"fold{args.fold}: n_test={len(test_ds)}", flush=True)

    device = get_device()
    state, source, epoch, best_val_f1 = load_state(args.checkpoint, device)

    backbone = Backbone(ENCODER_NAME, weights=None)
    head = CornHead(backbone.out_dim, num_classes)
    model = torch.nn.Sequential(backbone, head)
    model.load_state_dict(state)
    model.to(device).eval()

    criterion = build_loss("corn", num_classes)
    test_stats = run_epoch(model, test_loader, criterion, device, "corn", num_classes, optimizer=None)

    print(f"\nfold{args.fold} TEST results (checkpoint: {Path(args.checkpoint).name}, source={source}):")
    print(f"  accuracy = {test_stats['accuracy']:.4f}")
    print(f"  macro_f1 = {test_stats['macro_f1']:.4f}")
    print(f"  mae      = {test_stats['mae']:.4f}")
    print(f"  qwk      = {test_stats['qwk']:.4f}")

    if args.append:
        exp_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
        results_csv = EXP_DIR / exp_cfg["results_csv"]
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        row = {
            "config_id": CONFIG_ID, "encoder": ENCODER_NAME, "head": "corn",
            "weights": f"task2_manet_encoder_fold{args.fold + 1}", "fold": args.fold,
            "params_m": round(params_m, 3),
            "accuracy": round(test_stats["accuracy"], 4), "macro_f1": round(test_stats["macro_f1"], 4),
            "mae": round(test_stats["mae"], 4), "qwk": round(test_stats["qwk"], 4),
            "train_time_s": 0.0,  # trained elsewhere -- not timed by this script
        }
        append_result(results_csv, row)
        print(f"\nappended to {results_csv}"
              + (f" (NOTE: checkpoint stopped at epoch {epoch}/{recipe_cfg['epochs']}, "
                 f"not a full training run -- see note in results/notes if you add one)" if epoch is not None else ""))


if __name__ == "__main__":
    main()
