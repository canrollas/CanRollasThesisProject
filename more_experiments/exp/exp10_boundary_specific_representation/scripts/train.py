#!/usr/bin/env python3
"""Experiment 10, Phase A (Chapter 8): boundary-specific representation
learning for CORN.

logits = baseline_logits + delta_logits, where baseline_logits come from
the FROZEN, trained RGB MiT-B1+CORN baseline (task_3's own mit_b1__corn
config, loaded per fold -- not retrained from ImageNet, same convention
as exp5/6/7) and delta_logits come from three per-boundary residual
heads (see scripts/model.py). Three modes -- heads / attention /
contrastive -- share this file; only the head's internal wiring and
(for "contrastive") the loss change. The MiT-B1 backbone and the
baseline's own CornHead are frozen for the entire run in every mode.

Trained on the ORIGINAL, unmodified PIID staging photos
(datasets/wound-stage-classification/) -- same data, same 3-fold split
algorithm (src/splits.py's build_folds, same seed) as task_3's own
baseline, so results are directly, pairwise comparable without any new
data generation or leakage surface.

Safe to interrupt and resume, same pattern as exp5/6/7's train.py. Does
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

THIS_DIR = Path(__file__).resolve().parent  # exp/exp10_boundary_specific_representation/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.augmentations import ImageAugmentation  # noqa: E402
from src.dataset import StageDataset  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.model import build_model as build_rgb_only_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml, set_seed  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from engine import fit, run_epoch  # noqa: E402
from model import MODES, build_model, count_params_m, count_trainable_params_m, freeze_backbone, load_rgb_baseline  # noqa: E402

RGB_BASELINE_CONFIG_ID = "mit_b1__corn"  # task_3's own config_id for this architecture


def config_id(mode):
    return f"mit_b1__corn__boundary_{mode}"


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
    return train_loader, val_loader, test_loader


def verify_reproduces_baseline_subcomputation(model, rgb_checkpoint_path, probe_images, device, num_classes=4):
    """Loads the SAME checkpoint into task_3's own (unmodified) build_model
    and checks the frozen baseline sub-computation inside `model` matches
    it exactly, on real data -- catches a silently-wrong key split/load
    that a random-input unit test wouldn't necessarily catch. Does NOT
    check model's full logits against the reference, since by design
    (see model.py's docstring) they differ once the non-zero-initialized
    residual heads contribute a nonzero delta."""
    reference = build_rgb_only_model("mit_b1", None, "corn", num_classes=num_classes)
    state_dict = torch.load(rgb_checkpoint_path, map_location=device)
    reference.load_state_dict(state_dict, strict=True)
    reference.to(device).eval()

    model.eval()
    with torch.no_grad():
        reference_logits = reference(probe_images)
        reference_pooled = reference[0](probe_images)
        _logits, baseline_logits, _z_list = model(probe_images)
        pooled = model.backbone(probe_images).mean(dim=(2, 3))

    return {
        "model's pooled GAP vector matches reference backbone's GAP exactly":
            torch.equal(pooled, reference_pooled),
        "model's frozen baseline_logits sub-computation matches reference CornHead exactly":
            torch.equal(baseline_logits, reference_logits),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--contrastive-weight", type=float, default=None,
                   help="overrides configs/base.yaml's contrastive_weight; only used in --mode contrastive")
    p.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs, tiny subset -- sanity check")
    p.add_argument("--folds", default=None,
                   help="comma-separated fold indices to run, e.g. '1' or '0,2' -- default: all folds in order.")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = Path(args.data_root) if args.data_root else EXP_DIR / base_cfg["data_root"]

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
    contrastive_weight = (
        args.contrastive_weight if args.contrastive_weight is not None else base_cfg["contrastive_weight"]
    ) if args.mode == "contrastive" else 0.0
    cid = config_id(args.mode)

    print(f"device={device} config={cid} folds={len(folds)} max_epochs={max_epochs} patience={patience} "
          f"contrastive_weight={contrastive_weight}", flush=True)

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
            data_root, fold, base_cfg, batch_size, args.workers, args.smoke
        )

        model = build_model(
            base_cfg["encoder"], None, num_classes=num_classes,
            hidden_dim=base_cfg["hidden_dim"], mode=args.mode,
        )
        load_rgb_baseline(model, rgb_checkpoint_path, device)
        model.to(device)

        probe_images, _probe_targets = next(iter(val_loader))
        probe_images = probe_images.to(device)
        checks = verify_reproduces_baseline_subcomputation(model, rgb_checkpoint_path, probe_images, device, num_classes)
        for name, ok in checks.items():
            print(f"[fold{fold_idx}] [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        if not all(checks.values()):
            raise SystemExit(
                f"[fold{fold_idx}] STOP: the loaded model's frozen baseline sub-computation does not "
                f"match the reference checkpoint exactly -- see failed check(s) above. Not proceeding to training."
            )

        freeze_backbone(model)

        params_m = count_params_m(model)
        trainable_m = count_trainable_params_m(model)
        checkpoint_path = checkpoint_dir / f"{cid}_fold{fold_idx}.pt"
        resume_path = checkpoint_dir / f"{cid}_fold{fold_idx}.resume.pt"

        t0 = time.time()
        model, best_val_f1, diagnostics, backbone_unchanged, final_epoch = fit(
            model, train_loader, val_loader, device,
            max_epochs=max_epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
            eta_min_factor=base_cfg["eta_min_factor"],
            head="corn", num_classes=num_classes,
            log_prefix=f"[{cid} fold{fold_idx}]",
            checkpoint_path=checkpoint_path, resume_path=resume_path, patience=patience,
            contrastive_weight=contrastive_weight, contrastive_temperature=base_cfg["contrastive_temperature"],
        )
        train_time_s = time.time() - t0
        diagnostics_all[f"fold{fold_idx}"] = diagnostics

        criterion = build_loss("corn", num_classes)
        test_stats = run_epoch(model, test_loader, criterion, device, "corn", num_classes, optimizer=None)

        provenance = {
            "config_id": cid, "mode": args.mode, "fold": fold_idx,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rgb_baseline_checkpoint": str(rgb_checkpoint_path.relative_to(REPO_ROOT)),
            "pretraining_verification": checks,
            "phase": "A (MiT-B1 backbone + frozen baseline CornHead for the entire run)",
            "early_stopped_at_epoch": final_epoch, "max_epochs": max_epochs, "patience": patience,
            "backbone_and_baseline_head_unchanged_across_training": backbone_unchanged,
            "contrastive_weight": contrastive_weight,
            "params_m": round(params_m, 3), "trainable_params_m": round(trainable_m, 3),
            "lr": base_cfg["lr"], "seed": base_cfg["seed"],
        }
        with open(checkpoint_path.with_suffix(".provenance.json"), "w") as f:
            json.dump(provenance, f, indent=2)

        if not backbone_unchanged:
            print(f"[{cid} fold{fold_idx}] WARNING: frozen backbone/baseline-head logits moved during "
                  f"training -- freeze may not be working correctly, check provenance JSON.", file=sys.stderr)

        row = {
            "config_id": cid, "mode": args.mode, "encoder": base_cfg["encoder"], "head": "corn",
            "weights": "baseline_checkpoint", "fold": fold_idx,
            "params_m": round(params_m, 3), "trainable_params_m": round(trainable_m, 3),
            "accuracy": round(test_stats["accuracy"], 4), "macro_f1": round(test_stats["macro_f1"], 4),
            "mae": round(test_stats["mae"], 4), "qwk": round(test_stats["qwk"], 4),
            "early_stopped_at_epoch": final_epoch, "train_time_s": round(train_time_s, 1),
        }
        results_rows.append(row)
        print(f"[{cid} fold{fold_idx}] DONE test_acc={row['accuracy']:.4f} "
              f"test_f1={row['macro_f1']:.4f} test_qwk={row['qwk']:.4f} "
              f"(stopped epoch {final_epoch}/{max_epochs})", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config_id", "mode", "encoder", "head", "weights", "fold", "params_m", "trainable_params_m",
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
        for key in sorted(existing_by_key, key=lambda k: (k[0], k[1])):
            writer.writerow(existing_by_key[key])

    diagnostics_path = results_csv.with_name("exp10_diagnostics.json")
    existing_diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    existing_diagnostics.setdefault(cid, {})
    existing_diagnostics[cid].update(diagnostics_all)
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
