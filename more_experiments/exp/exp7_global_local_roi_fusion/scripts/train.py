#!/usr/bin/env python3
"""Experiment 7, Phase A (Chapter 8): global-local ROI feature fusion.

logits = fusion_head(concat(F_global, F_local)). F_global = GAP of the
FROZEN, trained RGB MiT-B1+CORN baseline's final-stage feature map (task_3's
own mit_b1__corn config, loaded per fold -- not retrained from ImageNet).
F_local = a projected, roi_align-pooled third-stage feature map over the
predicted wound ROI box (../results/roi_manifest.csv, from
generate_roi_manifest.py -- the leakage-free Task 1 ensemble, no staging
labels or ground-truth masks used to pick boxes). The fusion head is the
baseline's own CornHead widened to 640 inputs: 512 global columns+bias
copied exactly, 128 local columns zero-initialized, so the model
reproduces the baseline exactly before training (verified in
test_model.py, and again here against the real checkpoint). Only
local_proj + fusion_head train; the MiT-B1 backbone is frozen for the
entire Phase A run.

Safe to interrupt and resume, same pattern as exp5/exp6's train.py. Does
not write to task_3/ or any other more_experiments/exp/*/ -- only
./checkpoints and ./results here.
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

THIS_DIR = Path(__file__).resolve().parent  # exp/exp7_global_local_roi_fusion/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.losses import build_loss  # noqa: E402
from src.model import build_model as build_rgb_only_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml, set_seed  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from augmentations import PairedImageBoxAugmentation  # noqa: E402
from dataset import ROIStageDataset, load_roi_manifest  # noqa: E402
from engine import fit, run_epoch  # noqa: E402
from model import build_model, count_params_m, freeze_backbone, load_rgb_baseline  # noqa: E402

CONFIG_ID = "mit_b1__corn__global_local_roi_fusion"
RGB_BASELINE_CONFIG_ID = "mit_b1__corn"  # task_3's own config_id for this architecture


def make_loaders(data_root, fold, base_cfg, roi_lookup, batch_size, workers, smoke):
    aug = PairedImageBoxAugmentation(**base_cfg["augment"], size=base_cfg["image_size"])
    train_samples, val_samples, test_samples = fold["train"], fold["val"], fold["test"]

    if smoke:
        train_samples, val_samples, test_samples = train_samples[:8], val_samples[:4], test_samples[:4]

    train_ds = ROIStageDataset(train_samples, data_root, roi_lookup, base_cfg["image_size"], augment=aug)
    val_ds = ROIStageDataset(val_samples, data_root, roi_lookup, base_cfg["image_size"], augment=None)
    test_ds = ROIStageDataset(test_samples, data_root, roi_lookup, base_cfg["image_size"], augment=None)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def verify_reproduces_baseline(fusion_model, rgb_checkpoint_path, probe_rgb, probe_boxes, device, num_classes=4):
    """Loads the SAME checkpoint into task_3's own (unmodified) build_model
    and checks its output matches fusion_model's logits exactly, on real
    data -- catches a silently-wrong key split/load that a random-input
    unit test wouldn't necessarily catch."""
    reference = build_rgb_only_model("mit_b1", None, "corn", num_classes=num_classes)
    state_dict = torch.load(rgb_checkpoint_path, map_location=device)
    reference.load_state_dict(state_dict, strict=True)
    reference.to(device).eval()

    fusion_model.eval()
    with torch.no_grad():
        reference_logits = reference(probe_rgb)
        reference_f_global = reference[0](probe_rgb)
        logits, f_global = fusion_model(probe_rgb, probe_boxes)

    return {
        "fusion f_global matches reference backbone's GAP exactly": torch.equal(f_global, reference_f_global),
        "fusion logits match reference logits exactly (local columns zeroed at init)":
            torch.equal(logits, reference_logits),
        "fusion_head local columns are exactly zero":
            torch.equal(
                fusion_model.fusion_head.fc.weight[:, fusion_model.fusion_head.global_dim:],
                torch.zeros_like(fusion_model.fusion_head.fc.weight[:, fusion_model.fusion_head.global_dim:]),
            ),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs, tiny subset -- sanity check")
    p.add_argument("--folds", default=None,
                    help="comma-separated fold indices to run, e.g. '1' or '0,2' -- "
                         "default: all folds in order.")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = Path(args.data_root) if args.data_root else EXP_DIR / base_cfg["data_root"]
    roi_lookup = load_roi_manifest(EXP_DIR / base_cfg["roi_manifest_csv"])

    set_seed(base_cfg["seed"])
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    if args.smoke:
        folds = folds[:1]

    device = get_device()
    results_csv = EXP_DIR / base_cfg["results_csv"]
    checkpoint_dir = EXP_DIR / base_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = args.max_epochs if args.max_epochs is not None else (2 if args.smoke else base_cfg["max_epochs"])
    patience = args.patience if args.patience is not None else base_cfg["patience"]
    batch_size = args.batch if args.batch is not None else base_cfg["batch_size"]
    num_classes = len(base_cfg["classes"])

    print(f"device={device} config={CONFIG_ID} folds={len(folds)} max_epochs={max_epochs} patience={patience}",
          flush=True)

    results_rows = []
    diagnostics_all = {}
    wanted_folds = set(int(x) for x in args.folds.split(",")) if args.folds else None

    for fold_idx, fold in enumerate(folds):
        if wanted_folds is not None and fold_idx not in wanted_folds:
            continue
        rgb_checkpoint_path = (EXP_DIR / base_cfg["rgb_checkpoint_template"].format(fold=fold_idx)).resolve()
        if not rgb_checkpoint_path.is_file():
            raise SystemExit(
                f"\nBLOCKED: RGB baseline checkpoint not found for fold {fold_idx}:\n"
                f"  {rgb_checkpoint_path}\n"
                f"This experiment requires task_3's own trained mit_b1__corn checkpoint per fold "
                f"(config_id={RGB_BASELINE_CONFIG_ID!r}) -- not retrained here. Do not substitute "
                f"a checkpoint trained on another fold."
            )
        print(f"[fold{fold_idx}] loading RGB baseline from {rgb_checkpoint_path}", flush=True)

        train_loader, val_loader, test_loader = make_loaders(
            data_root, fold, base_cfg, roi_lookup, batch_size, args.workers, args.smoke
        )

        model = build_model(
            base_cfg["encoder"], None, num_classes=num_classes,
            roi_output_size=base_cfg["roi_output_size"], local_dim=base_cfg["local_dim"],
        )
        load_rgb_baseline(model, rgb_checkpoint_path, device)
        model.to(device)

        from engine import _boxes_to_list  # noqa: E402 (local import, avoids a module-level cycle)
        probe_rgb, probe_boxes_raw, _, _ = next(iter(val_loader))
        probe_rgb = probe_rgb.to(device)
        probe_boxes = _boxes_to_list(probe_boxes_raw.to(device))
        checks = verify_reproduces_baseline(model, rgb_checkpoint_path, probe_rgb, probe_boxes, device, num_classes)
        for name, ok in checks.items():
            print(f"[fold{fold_idx}] [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        if not all(checks.values()):
            raise SystemExit(
                f"[fold{fold_idx}] STOP: the initialized fusion model does not reproduce the RGB "
                f"baseline's logits exactly -- see failed check(s) above. Not proceeding to training."
            )

        freeze_backbone(model)

        params_m = count_params_m(model)
        trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        checkpoint_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.pt"
        resume_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.resume.pt"

        t0 = time.time()
        model, best_val_f1, diagnostics, backbone_unchanged, final_epoch = fit(
            model, train_loader, val_loader, device,
            max_epochs=max_epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
            eta_min_factor=base_cfg["eta_min_factor"],
            head="corn", num_classes=num_classes,
            log_prefix=f"[{CONFIG_ID} fold{fold_idx}]",
            checkpoint_path=checkpoint_path, resume_path=resume_path, patience=patience,
        )
        train_time_s = time.time() - t0
        diagnostics_all[f"fold{fold_idx}"] = diagnostics

        criterion = build_loss("corn", num_classes)
        test_stats = run_epoch(model, test_loader, criterion, device, "corn", num_classes, optimizer=None)

        n_failed_localization = sum(1 for _, v in roi_lookup.items() if v["fold"] == fold_idx and not v["detected"])

        provenance = {
            "config_id": CONFIG_ID, "fold": fold_idx,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rgb_baseline_checkpoint": str(rgb_checkpoint_path.relative_to(REPO_ROOT)),
            "pretraining_verification": checks,
            "phase": "A (MiT-B1 backbone frozen for the entire run)",
            "early_stopped_at_epoch": final_epoch, "max_epochs": max_epochs, "patience": patience,
            "backbone_f_global_unchanged_across_training": backbone_unchanged,
            "n_failed_localizations_in_fold_test_split": n_failed_localization,
            "params_m": round(params_m, 3), "trainable_params_m": round(trainable_m, 3),
            "lr": base_cfg["lr"], "seed": base_cfg["seed"],
        }
        with open(checkpoint_path.with_suffix(".provenance.json"), "w") as f:
            json.dump(provenance, f, indent=2)

        if not backbone_unchanged:
            print(f"[{CONFIG_ID} fold{fold_idx}] WARNING: backbone f_global moved during training "
                  f"-- freeze may not be working correctly, check provenance JSON.", file=sys.stderr)

        row = {
            "config_id": CONFIG_ID, "encoder": base_cfg["encoder"], "head": "corn",
            "weights": "baseline_checkpoint", "fold": fold_idx,
            "params_m": round(params_m, 3), "trainable_params_m": round(trainable_m, 3),
            "accuracy": round(test_stats["accuracy"], 4), "macro_f1": round(test_stats["macro_f1"], 4),
            "mae": round(test_stats["mae"], 4), "qwk": round(test_stats["qwk"], 4),
            "early_stopped_at_epoch": final_epoch, "train_time_s": round(train_time_s, 1),
        }
        results_rows.append(row)
        print(f"[{CONFIG_ID} fold{fold_idx}] DONE test_acc={row['accuracy']:.4f} "
              f"test_f1={row['macro_f1']:.4f} test_qwk={row['qwk']:.4f} "
              f"(stopped epoch {final_epoch}/{max_epochs})", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config_id", "encoder", "head", "weights", "fold", "params_m", "trainable_params_m",
                  "accuracy", "macro_f1", "mae", "qwk", "early_stopped_at_epoch", "train_time_s"]

    existing_by_key = {}
    if results_csv.exists():
        with open(results_csv, newline="") as f:
            for row in csv.DictReader(f):
                existing_by_key[(row["config_id"], int(row["fold"]))] = row
    for row in results_rows:
        existing_by_key[(row["config_id"], row["fold"])] = row

    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(existing_by_key):
            writer.writerow(existing_by_key[key])

    diagnostics_path = results_csv.with_name("exp7_diagnostics.json")
    existing_diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    existing_diagnostics.setdefault(CONFIG_ID, {})
    existing_diagnostics[CONFIG_ID].update(diagnostics_all)
    with open(diagnostics_path, "w") as f:
        json.dump(existing_diagnostics, f, indent=2)

    print(f"wrote {results_csv}")
    print(f"wrote {diagnostics_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
