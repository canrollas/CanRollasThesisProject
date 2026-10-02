#!/usr/bin/env python3
"""Stage-2 intra-wound tissue segmentation ablation runner.

Trains + evaluates the 10 top-ranked (architecture, encoder, weights)
configs from configs/selected_configs.json (picked from the paper's 37-config
Table 5 benchmark) under 3-fold cross-validation on this project's
wound-tissue-segmentation dataset. For every (config, fold) pair not already
present in the results CSV, trains on that fold's train/val split, evaluates
on the held-out test fold, and appends one row to the results CSV. Safe to
interrupt and re-run (already-completed rows are skipped), so the grid can be
worked through across multiple Colab sessions.
"""
import argparse
import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR))

from src.augmentations import PairedAugmentation  # noqa: E402
from src.dataset import TissueDataset, load_color_mapping  # noqa: E402
from src.efficiency import count_params_m  # noqa: E402
from src.engine import fit, run_epoch  # noqa: E402
from src.losses import CombinedLoss  # noqa: E402
from src.model import build_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import (  # noqa: E402
    append_failure, append_result, get_device, load_completed_keys, load_yaml, set_seed, write_summary,
)
from src.visualize import save_sample_panel  # noqa: E402


def make_loaders(data_root, fold, base_cfg, colors, batch_size, workers, smoke):
    aug = PairedAugmentation(**base_cfg["augment"])
    train_files, val_files, test_files = fold["train"], fold["val"], fold["test"]

    if smoke:
        train_files, val_files, test_files = train_files[:8], val_files[:4], test_files[:4]

    train_ds = TissueDataset(train_files, data_root, base_cfg["image_size"], colors, transform=aug)
    val_ds = TissueDataset(val_files, data_root, base_cfg["image_size"], colors, transform=None)
    test_ds = TissueDataset(test_files, data_root, base_cfg["image_size"], colors, transform=None)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None, help="defaults to <repo>/datasets/wound-tissue-segmentation")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--configs", default=None,
                    help="comma-separated config_id allowlist (from configs/selected_configs.json), "
                         "e.g. 'unet__mit_b2__imagenet,manet__mit_b2__imagenet' -- default: all 10")
    p.add_argument("--smoke", action="store_true", help="1 config, 1 fold, 2 epochs, tiny subset")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(TASK_DIR / "configs" / "base.yaml")
    resolved_path = TASK_DIR / "configs" / "selected_configs.json"
    if not resolved_path.exists():
        raise SystemExit("configs/selected_configs.json not found — run build_selected_configs.py first")
    grid = json.loads(resolved_path.read_text())

    if args.configs is not None:
        wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
        by_id = {c["config_id"]: c for c in grid}
        missing = [c for c in wanted if c not in by_id]
        if missing:
            raise SystemExit(f"--configs: unknown config_id(s) not in selected_configs.json: {missing}")
        grid = [by_id[c] for c in wanted]
    if args.smoke:
        grid = grid[:1]

    repo_root = TASK_DIR.parent
    data_root = Path(args.data_root) if args.data_root else repo_root / "datasets" / "wound-tissue-segmentation"
    colors = load_color_mapping(data_root)

    set_seed(base_cfg["seed"])
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    if args.smoke:
        folds = folds[:1]

    device = get_device()
    results_csv = TASK_DIR / base_cfg["results_csv"]
    failures_csv = results_csv.with_name("ablation_failures.csv")
    checkpoint_dir = TASK_DIR / base_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(results_csv)

    epochs = args.epochs if args.epochs is not None else (2 if args.smoke else base_cfg["epochs"])
    batch_size = args.batch if args.batch is not None else base_cfg["batch_size"]
    num_classes = len(base_cfg["classes"])

    print(f"device={device} configs={len(grid)} folds={len(folds)} epochs={epochs}", flush=True)

    for cfg in grid:
        for fold_idx, fold in enumerate(folds):
            if (cfg["config_id"], str(fold_idx)) in completed:
                continue

            model = None
            try:
                train_loader, val_loader, test_loader = make_loaders(
                    data_root, fold, base_cfg, colors, batch_size, args.workers, args.smoke
                )

                model = build_model(cfg["architecture"], cfg["encoder"], cfg["weights"], num_classes=num_classes)
                params_m = count_params_m(model)
                checkpoint_path = checkpoint_dir / f"{cfg['config_id']}_fold{fold_idx}.pt"
                resume_path = checkpoint_dir / f"{cfg['config_id']}_fold{fold_idx}.resume.pt"

                t0 = time.time()
                model, _ = fit(
                    model, train_loader, val_loader, device,
                    epochs=epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
                    eta_min_factor=base_cfg["eta_min_factor"],
                    ce_weight=base_cfg["loss"]["ce_weight"], dice_weight=base_cfg["loss"]["dice_weight"],
                    num_classes=num_classes,
                    log_prefix=f"[{cfg['config_id']} fold{fold_idx}]",
                    checkpoint_path=checkpoint_path,
                    resume_path=resume_path,
                )
                train_time_s = time.time() - t0

                criterion = CombinedLoss(num_classes=num_classes,
                                          ce_weight=base_cfg["loss"]["ce_weight"],
                                          dice_weight=base_cfg["loss"]["dice_weight"])
                test_stats = run_epoch(model, test_loader, criterion, device, optimizer=None,
                                        num_classes=num_classes)

                try:
                    sample_idx = random.randrange(len(test_loader.dataset))
                    sample_image, sample_gt = test_loader.dataset[sample_idx]
                    model.eval()
                    with torch.no_grad():
                        logits = model(sample_image.unsqueeze(0).to(device))
                        pred_mask = logits.argmax(dim=1).squeeze(0).cpu()
                    sample_path = results_csv.parent / "samples" / f"{cfg['config_id']}_fold{fold_idx}.png"
                    save_sample_panel(sample_image, sample_gt, pred_mask, colors, sample_path)
                except Exception as exc:  # noqa: BLE001 - a bad sample image shouldn't kill the whole grid
                    print(f"[{cfg['config_id']} fold{fold_idx}] sample panel failed: {exc}", flush=True)

                row = {
                    "config_id": cfg["config_id"],
                    "architecture": cfg["architecture"],
                    "encoder": cfg["encoder"],
                    "weights": cfg["weights"],
                    "fold": fold_idx,
                    "params_m": round(params_m, 3),
                    "miou": round(test_stats["miou"], 4),
                    "mdice": round(test_stats["mdice"], 4),
                    "iou_granulation": round(test_stats["iou_per_class"][0], 4),
                    "iou_slough": round(test_stats["iou_per_class"][1], 4),
                    "iou_necrosis": round(test_stats["iou_per_class"][2], 4),
                    "dice_granulation": round(test_stats["dice_per_class"][0], 4),
                    "dice_slough": round(test_stats["dice_per_class"][1], 4),
                    "dice_necrosis": round(test_stats["dice_per_class"][2], 4),
                    "train_time_s": round(train_time_s, 1),
                }
                append_result(results_csv, row)
                write_summary(results_csv, results_csv.with_name("ablation_summary.csv"))
                print(f"[{cfg['config_id']} fold{fold_idx}] DONE test_miou={row['miou']:.4f}", flush=True)

            except Exception as exc:  # noqa: BLE001 - one bad config must not take the whole grid down
                traceback.print_exc()
                print(f"[{cfg['config_id']} fold{fold_idx}] FAILED: {exc}", flush=True)
                append_failure(failures_csv, {
                    "config_id": cfg["config_id"],
                    "architecture": cfg["architecture"],
                    "encoder": cfg["encoder"],
                    "weights": cfg["weights"],
                    "fold": fold_idx,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })

            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
