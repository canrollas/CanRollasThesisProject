#!/usr/bin/env python3
"""Experiment 5, Phase 1 (Chapter 8): baseline-preserving residual late fusion.

z_final = z_rgb + delta_z. z_rgb comes from the FROZEN, trained RGB
MiT-B1+CORN baseline (task_3's own mit_b1__corn config, loaded per fold --
NOT ImageNet-only, unlike exp-4). delta_z comes from a small residual MLP
(tissue CNN + fusion head) whose final layer is zero-initialized, so the
model reproduces the baseline exactly before training. Only the tissue
CNN + residual head train; the RGB branch (backbone + CORN head) is
frozen for the entire run.

Two verifications run before any training is trusted:
  1. scripts/test_residual_model.py (architecture-level, run separately,
     doesn't need a real checkpoint).
  2. HERE, against the real loaded baseline checkpoint and real data: a
     batch is run through the frozen RGB branch inside this fusion model
     AND through task_3's own build_model with the same checkpoint
     loaded, and the two must match exactly. If they don't, this script
     stops immediately rather than training on top of a silently broken
     baseline load.

Uses exp-4's dataset (same leakage-safe tissue predictions, same verified
PIID fold split) and exp-4's TissueCNN, reused unchanged.

Safe to interrupt and resume, same pattern as the other exp/*/scripts/train.py.
Does not write to task_3/, exp3_tissue_channels/, or exp4_late_fusion/ --
only ./checkpoints and ./results here.
"""
import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent  # exp/exp5_residual_fusion/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.augmentations import ImageAugmentation  # noqa: E402
from src.model import build_model as build_rgb_only_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml, set_seed  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from dataset import TwoStreamStageDataset  # noqa: E402
from engine import fit, run_epoch  # noqa: E402
from model import build_model, load_rgb_baseline  # noqa: E402
from model import count_params_m  # noqa: E402

CONFIG_IDS = {
    "rgb_tissue": "mit_b1__corn__residual_latefusion",
    "tissue_only": "mit_b1__corn__tissue_only_residual",
}
RGB_BASELINE_CONFIG_ID = "mit_b1__corn"  # task_3's own config_id for this architecture


def make_loaders(data_root, fold, base_cfg, batch_size, workers, smoke):
    aug = ImageAugmentation(**base_cfg["augment"])
    train_samples, val_samples, test_samples = fold["train"], fold["val"], fold["test"]

    if smoke:
        train_samples, val_samples, test_samples = train_samples[:8], val_samples[:4], test_samples[:4]

    train_ds = TwoStreamStageDataset(train_samples, data_root, base_cfg["image_size"], transform=aug)
    val_ds = TwoStreamStageDataset(val_samples, data_root, base_cfg["image_size"], transform=None)
    test_ds = TwoStreamStageDataset(test_samples, data_root, base_cfg["image_size"], transform=None)

    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def verify_reproduces_baseline(fusion_model, rgb_checkpoint_path, probe_rgb, device, num_classes=4):
    """Loads the SAME checkpoint into task_3's own (unmodified) build_model
    and checks its output matches fusion_model's z_rgb exactly, on real
    data. Catches a silently-wrong key split/load in model.load_rgb_baseline
    that unit tests with random inputs wouldn't necessarily catch."""
    reference = build_rgb_only_model("mit_b1", None, "corn", num_classes=num_classes)
    state_dict = torch.load(rgb_checkpoint_path, map_location=device)
    reference.load_state_dict(state_dict, strict=True)
    reference.to(device).eval()

    fusion_model.eval()
    with torch.no_grad():
        reference_logits = reference(probe_rgb)
        dummy_tissue = torch.zeros(probe_rgb.shape[0], 5, *probe_rgb.shape[2:], device=device)
        z_final, z_rgb, delta_z = fusion_model(probe_rgb, dummy_tissue)

    checks = {
        "fusion z_rgb matches reference exactly": torch.equal(z_rgb, reference_logits),
        "fusion z_final matches reference exactly (delta_z==0 at init)": torch.equal(z_final, reference_logits),
        "delta_z is exactly zero at init": torch.equal(delta_z, torch.zeros_like(delta_z)),
    }
    return checks


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
                         "default: all folds in order. Useful when only some folds' "
                         "RGB baseline checkpoints exist yet.")
    p.add_argument("--variant", default="tissue_only", choices=["rgb_tissue", "tissue_only"],
                    help="'tissue_only' (default, new): residual head sees ONLY tissue "
                         "features. 'rgb_tissue' (original): residual head sees RGB+tissue "
                         "concatenated -- diagnose_residual.py found this variant's correction "
                         "doesn't actually depend on tissue content (cosine(real,shuffled "
                         "tissue) ~1.0 at convergence); kept for reference, not retrained by "
                         "default.")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    CONFIG_ID = CONFIG_IDS[args.variant]

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
                f"(config_id={RGB_BASELINE_CONFIG_ID!r}) -- it is NOT retrained here, per the brief's "
                f"'do not retrain the RGB baseline from ImageNet' requirement. Train it with task_3's "
                f"own, unmodified ablation script first:\n"
                f"  cd task_3 && python run_ablation.py --configs {RGB_BASELINE_CONFIG_ID}\n"
                f"(fast: ~90s/fold per task_3/results/task_3_outputs/results/ablation_results.csv)."
            )

        print(f"[fold{fold_idx}] loading RGB baseline from {rgb_checkpoint_path}", flush=True)

        train_loader, val_loader, test_loader = make_loaders(
            data_root, fold, base_cfg, batch_size, args.workers, args.smoke
        )

        model = build_model(base_cfg["encoder"], base_cfg["weights"], num_classes=num_classes, variant=args.variant)
        load_rgb_baseline(model, rgb_checkpoint_path, device)
        model.to(device)

        probe_rgb, _, _ = next(iter(val_loader))
        probe_rgb = probe_rgb.to(device)
        checks = verify_reproduces_baseline(model, rgb_checkpoint_path, probe_rgb, device, num_classes)
        for name, ok in checks.items():
            status = "PASS" if ok else "FAIL"
            print(f"[fold{fold_idx}] [{status}] {name}", flush=True)
        if not all(checks.values()):
            raise SystemExit(
                f"[fold{fold_idx}] STOP: the initialized residual-fusion model does not reproduce "
                f"the RGB baseline's logits exactly -- see failed check(s) above. Not proceeding to "
                f"training on top of a broken baseline load."
            )

        params_m = count_params_m(model)
        checkpoint_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.pt"
        resume_path = checkpoint_dir / f"{CONFIG_ID}_fold{fold_idx}.resume.pt"

        t0 = time.time()
        model, best_val_f1, diagnostics, rgb_unchanged, final_epoch = fit(
            model, train_loader, val_loader, device,
            max_epochs=max_epochs, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
            eta_min_factor=base_cfg["eta_min_factor"],
            head="corn", num_classes=num_classes,
            log_prefix=f"[{CONFIG_ID} fold{fold_idx}]",
            checkpoint_path=checkpoint_path, resume_path=resume_path, patience=patience,
        )
        train_time_s = time.time() - t0
        diagnostics_all[f"fold{fold_idx}"] = diagnostics

        # Residual-disabled inference check: force delta_z=0 post-training and
        # confirm it reproduces z_rgb -- true by construction (z_final = z_rgb +
        # delta_z), checked explicitly anyway per the brief.
        model.eval()
        with torch.no_grad():
            rgb_batch, tissue_batch, _ = next(iter(test_loader))
            rgb_batch, tissue_batch = rgb_batch.to(device), tissue_batch.to(device)
            z_final, z_rgb, delta_z = model(rgb_batch, tissue_batch)
            z_final_disabled = z_rgb + torch.zeros_like(delta_z)
            residual_disabled_reproduces_rgb = torch.equal(z_final_disabled, z_rgb)
            nonzero_corrections_after_training = delta_z.abs().sum().item() > 0

        from src.losses import build_loss  # noqa: E402 (local import, same pattern as elsewhere in this file)
        criterion = build_loss("corn", num_classes)
        test_stats = run_epoch(model, test_loader, criterion, device, "corn", num_classes, optimizer=None)

        provenance = {
            "config_id": CONFIG_ID, "variant": args.variant, "fold": fold_idx,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rgb_baseline_checkpoint": str(rgb_checkpoint_path.relative_to(REPO_ROOT)),
            "pretraining_verification": checks,
            "phase": "1 (RGB branch frozen for the entire run)",
            "early_stopped_at_epoch": final_epoch, "max_epochs": max_epochs, "patience": patience,
            "rgb_logits_unchanged_across_training": rgb_unchanged,
            "residual_disabled_check_reproduces_rgb": residual_disabled_reproduces_rgb,
            "nonzero_corrections_after_training": nonzero_corrections_after_training,
            "final_delta_z_norm_mean_test_batch": round(delta_z.norm(dim=1).mean().item(), 4),
            "params_m": round(params_m, 3), "lr": base_cfg["lr"], "seed": base_cfg["seed"],
        }
        with open(checkpoint_path.with_suffix(".provenance.json"), "w") as f:
            json.dump(provenance, f, indent=2)

        if not rgb_unchanged:
            print(f"[{CONFIG_ID} fold{fold_idx}] WARNING: RGB branch logits moved during training "
                  f"-- freeze may not be working correctly, check provenance JSON.", file=sys.stderr)

        row = {
            "config_id": CONFIG_ID, "encoder": base_cfg["encoder"], "head": "corn",
            "weights": "baseline_checkpoint", "fold": fold_idx, "params_m": round(params_m, 3),
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

    import csv
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config_id", "encoder", "head", "weights", "fold", "params_m",
                  "accuracy", "macro_f1", "mae", "qwk", "early_stopped_at_epoch", "train_time_s"]

    # Merge with whatever's already on disk (keyed by (config_id, fold) --
    # NOT fold alone, since multiple variants share this one results file)
    # instead of overwriting -- folds/variants may be trained incrementally.
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

    diagnostics_path = results_csv.with_name("exp5_diagnostics.json")
    existing_diagnostics = {}
    if diagnostics_path.exists():
        existing_diagnostics = json.loads(diagnostics_path.read_text())
    existing_diagnostics.setdefault(CONFIG_ID, {})
    existing_diagnostics[CONFIG_ID].update(diagnostics_all)
    with open(diagnostics_path, "w") as f:
        json.dump(existing_diagnostics, f, indent=2)

    print(f"wrote {results_csv}")
    print(f"wrote {results_csv.with_name('exp5_diagnostics.json')}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
