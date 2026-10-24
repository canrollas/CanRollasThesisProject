#!/usr/bin/env python3
"""Chapter 8, exp-6 follow-up: does tissue_statistics_only carry a signal
that is actually COMPLEMENTARY to the RGB MiT-B1+CORN baseline, i.e. does
it tend to be right exactly where RGB is wrong -- or is its (weaker)
signal just a subset of what RGB already gets right?

Pure inference / analysis script: loads the ALREADY-TRAINED checkpoints
for both models (no new training, no checkpoint or results-file changes)
and matches their per-image test-set predictions by filename stem (both
datasets are built from the same seed=42 stratified 3-fold split over
the same 1091 images -- task_3/src/splits.py build_folds -- so fold i's
test set is the exact same 1,091/3 images regardless of which dataset
root's directory listing produced it; verified by stem-set equality
below, not assumed).

Reports, per fold and pooled:
  1. The 2x2 (RGB correct/wrong) x (Tissue correct/wrong) contingency table.
  2. The RGB-wrong / Tissue-correct cases, broken down by true stage and by
     what each model actually predicted.
  3. tissue_statistics_only's majority-class baseline and per-class F1.
  4. P(tissue correct | RGB wrong).
  5. Error overlap between the two models.
  6. Fold 1's residual-fusion model: RGB-only vs real-tissue-fusion vs
     shuffled-tissue-fusion metrics on the same test set (sanity-checks
     against the already-recorded exp6_results.csv row for fold 1).
  7. An ORACLE selection accuracy (pick whichever model is correct, using
     the true label to choose) -- reported explicitly as a theoretical
     upper bound on complementarity, not as an achievable selection model.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent  # exp/exp6_tissue_statistics/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.dataset import StageDataset  # noqa: E402
from src.metrics import compute_metrics, predict_labels  # noqa: E402
from src.model import build_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from dataset import TissueStatsOnlyDataset, TissueStatsResidualDataset, load_feature_table  # noqa: E402
from model import build_residual_model, build_stats_only_model  # noqa: E402

CLASS_ORDER = ["stage_1", "stage_2", "stage_3", "stage_4"]
RGB_CKPT_DIR = REPO_ROOT / "task_3" / "results" / "task_3_outputs" / "checkpoints"
RGB_DATA_ROOT = REPO_ROOT / "datasets" / "wound-stage-classification"


def stem_of(rel_path):
    return Path(rel_path).stem


@torch.no_grad()
def rgb_predictions(fold_idx, base_cfg, device):
    folds = build_folds(RGB_DATA_ROOT, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    test_samples = folds[fold_idx]["test"]
    ds = StageDataset(test_samples, RGB_DATA_ROOT, base_cfg["image_size"], transform=None)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_model("mit_b1", None, "corn", num_classes=4).to(device)
    model.load_state_dict(torch.load(RGB_CKPT_DIR / f"mit_b1__corn_fold{fold_idx}.pt", map_location=device))
    model.eval()

    preds, trues = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        preds.extend(predict_labels(logits, "corn").cpu().tolist())
        trues.extend(targets.tolist())

    stems = [stem_of(rel) for rel, _ in test_samples]
    return {s: (t, p) for s, t, p in zip(stems, trues, preds)}


@torch.no_grad()
def tissue_only_predictions(fold_idx, base_cfg, data_root, feature_lookup, device):
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    test_samples = folds[fold_idx]["test"]
    train_samples = folds[fold_idx]["train"]
    ds = TissueStatsOnlyDataset(test_samples, feature_lookup)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_stats_only_model(num_classes=4).to(device)
    ckpt = EXP_DIR / "checkpoints" / f"mit_b1__corn__tissue_statistics_only_fold{fold_idx}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    preds, trues = [], []
    for stats, targets in loader:
        logits = model(stats.to(device))
        preds.extend(predict_labels(logits, "corn").cpu().tolist())
        trues.extend(targets.tolist())

    stems = [stem_of(rel) for rel, _ in test_samples]
    train_label_counts = Counter(label for _, label in train_samples)
    return {s: (t, p) for s, t, p in zip(stems, trues, preds)}, train_label_counts


@torch.no_grad()
def fold1_residual_shuffle_metrics(base_cfg, data_root, feature_lookup, device):
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    test_samples = folds[1]["test"]
    ds = TissueStatsResidualDataset(test_samples, data_root, feature_lookup, base_cfg.get("image_size", 224), transform=None)

    rgbs, stats, labels = [], [], []
    for i in range(len(ds)):
        r, s, label = ds[i]
        rgbs.append(r)
        stats.append(s)
        labels.append(label)
    rgb = torch.stack(rgbs).to(device)
    stats = torch.stack(stats).to(device)
    labels_t = torch.tensor(labels)

    model = build_residual_model("mit_b1", None, num_classes=4).to(device)
    ckpt = EXP_DIR / "checkpoints" / "mit_b1__corn__tissue_statistics_residual_fusion_fold1.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    torch.manual_seed(0)
    perm = torch.randperm(stats.shape[0])
    z_final_real, z_rgb, delta_z_real = model(rgb, stats)
    delta_z_shuffled = model.residual_head(stats[perm])
    z_final_shuffled = z_rgb + delta_z_shuffled

    y_true = labels_t.numpy()
    results = {}
    for name, logits in [("rgb_only", z_rgb), ("real_fusion", z_final_real), ("shuffled_fusion", z_final_shuffled)]:
        pred = predict_labels(logits, "corn").cpu().numpy()
        m = compute_metrics(y_true, pred, num_classes=4)
        results[name] = m
    return results, len(test_samples)


def contingency(rgb_map, tissue_map):
    assert set(rgb_map) == set(tissue_map), "stem sets differ between RGB and tissue test splits"
    rows = []
    for stem in rgb_map:
        true_rgb, pred_rgb = rgb_map[stem]
        true_tis, pred_tis = tissue_map[stem]
        assert true_rgb == true_tis, f"{stem}: label mismatch {true_rgb} vs {true_tis}"
        rows.append((stem, true_rgb, pred_rgb, pred_tis))
    return rows


def print_fold_report(fold_idx, rows, train_label_counts):
    n = len(rows)
    groups = {"both_correct": [], "rgb_only_correct": [], "tissue_only_correct": [], "both_wrong": []}
    for stem, true, pred_rgb, pred_tis in rows:
        rgb_ok, tis_ok = pred_rgb == true, pred_tis == true
        if rgb_ok and tis_ok:
            groups["both_correct"].append((stem, true, pred_rgb, pred_tis))
        elif rgb_ok and not tis_ok:
            groups["rgb_only_correct"].append((stem, true, pred_rgb, pred_tis))
        elif not rgb_ok and tis_ok:
            groups["tissue_only_correct"].append((stem, true, pred_rgb, pred_tis))
        else:
            groups["both_wrong"].append((stem, true, pred_rgb, pred_tis))

    print(f"\n{'='*70}\nFOLD {fold_idx}  (n={n})\n{'='*70}")
    print(f"{'group':<22}{'n':>6}{'%':>8}")
    for name in ["both_correct", "rgb_only_correct", "tissue_only_correct", "both_wrong"]:
        g = groups[name]
        print(f"{name:<22}{len(g):>6}{100*len(g)/n:>7.1f}%")

    rgb_correct_n = len(groups["both_correct"]) + len(groups["rgb_only_correct"])
    rgb_wrong_n = n - rgb_correct_n
    tissue_correct_n = len(groups["both_correct"]) + len(groups["tissue_only_correct"])
    tissue_wrong_n = n - tissue_correct_n

    p_tissue_correct_given_rgb_wrong = (
        len(groups["tissue_only_correct"]) / rgb_wrong_n if rgb_wrong_n else float("nan")
    )
    overlap_of_rgb_errors = len(groups["both_wrong"]) / rgb_wrong_n if rgb_wrong_n else float("nan")
    overlap_of_tissue_errors = len(groups["both_wrong"]) / tissue_wrong_n if tissue_wrong_n else float("nan")

    print(f"\nRGB accuracy (this analysis)    = {rgb_correct_n}/{n} = {rgb_correct_n/n:.4f}")
    print(f"Tissue accuracy (this analysis) = {tissue_correct_n}/{n} = {tissue_correct_n/n:.4f}")
    print(f"P(tissue correct | RGB wrong)   = {len(groups['tissue_only_correct'])}/{rgb_wrong_n} = {p_tissue_correct_given_rgb_wrong:.4f}")
    print(f"P(RGB correct | tissue wrong)   = {len(groups['rgb_only_correct'])}/{tissue_wrong_n} = {len(groups['rgb_only_correct'])/tissue_wrong_n if tissue_wrong_n else float('nan'):.4f}")
    print(f"error overlap: fraction of RGB's {rgb_wrong_n} errors tissue ALSO misses    = {overlap_of_rgb_errors:.4f}")
    print(f"error overlap: fraction of tissue's {tissue_wrong_n} errors RGB ALSO misses = {overlap_of_tissue_errors:.4f}")

    oracle_correct_n = n - len(groups["both_wrong"])
    print(f"\n[THEORETICAL UPPER BOUND ONLY -- not an achievable model, uses true labels to pick]")
    print(f"oracle selection accuracy (correct if EITHER model is correct) = {oracle_correct_n}/{n} = {oracle_correct_n/n:.4f}")

    # majority-class baseline + per-class F1 for tissue_statistics_only
    majority_class = train_label_counts.most_common(1)[0][0]
    y_true = np.array([t for _, t, _, _ in rows])
    y_pred_tissue = np.array([pt for _, _, _, pt in rows])
    y_pred_majority = np.full_like(y_true, majority_class)
    maj_acc = (y_true == y_pred_majority).mean()
    maj_f1_per_class = f1_score(y_true, y_pred_majority, average=None, labels=[0, 1, 2, 3], zero_division=0)
    tis_f1_per_class = f1_score(y_true, y_pred_tissue, average=None, labels=[0, 1, 2, 3], zero_division=0)

    print(f"\ntissue_statistics_only vs majority-class baseline (majority class = {CLASS_ORDER[majority_class]}, from this fold's train split):")
    print(f"  majority-baseline accuracy = {maj_acc:.4f}")
    print(f"  {'class':<10}{'majority F1':>14}{'tissue-only F1':>16}")
    for c in range(4):
        print(f"  {CLASS_ORDER[c]:<10}{maj_f1_per_class[c]:>14.4f}{tis_f1_per_class[c]:>16.4f}")

    if groups["tissue_only_correct"]:
        print(f"\nRGB-wrong / Tissue-correct cases ({len(groups['tissue_only_correct'])}):")
        print(f"  {'stem':<24}{'true':<10}{'rgb_pred':<10}{'tissue_pred':<12}")
        for stem, true, pred_rgb, pred_tis in groups["tissue_only_correct"]:
            print(f"  {stem:<24}{CLASS_ORDER[true]:<10}{CLASS_ORDER[pred_rgb]:<10}{CLASS_ORDER[pred_tis]:<12}")
        true_stage_counts = Counter(CLASS_ORDER[t] for _, t, _, _ in groups["tissue_only_correct"])
        rgb_mistake_counts = Counter(f"{CLASS_ORDER[t]}->{CLASS_ORDER[p]}" for _, t, p, _ in groups["tissue_only_correct"])
        print(f"  true-stage distribution: {dict(true_stage_counts)}")
        print(f"  RGB's mistake pattern (true->rgb_pred): {dict(rgb_mistake_counts)}")

    return groups, rgb_correct_n, tissue_correct_n, n


def main():
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = EXP_DIR / base_cfg["data_root"]
    feature_lookup = load_feature_table(EXP_DIR / base_cfg["feature_csv"])
    device = get_device()
    print(f"device={device}")
    print(f"RGB data root:    {RGB_DATA_ROOT}")
    print(f"Tissue data root: {data_root}")

    all_rows = []
    pooled_both_wrong = pooled_rgb_wrong = pooled_tissue_wrong = pooled_tissue_correct_given_rgb_wrong = 0
    pooled_n = 0

    for fold_idx in range(3):
        rgb_map = rgb_predictions(fold_idx, base_cfg, device)
        tissue_map, train_label_counts = tissue_only_predictions(fold_idx, base_cfg, data_root, feature_lookup, device)
        rows = contingency(rgb_map, tissue_map)
        groups, rgb_correct_n, tissue_correct_n, n = print_fold_report(fold_idx, rows, train_label_counts)
        all_rows.extend(rows)

        rgb_wrong_n = n - rgb_correct_n
        tissue_wrong_n = n - tissue_correct_n
        pooled_both_wrong += len(groups["both_wrong"])
        pooled_rgb_wrong += rgb_wrong_n
        pooled_tissue_wrong += tissue_wrong_n
        pooled_tissue_correct_given_rgb_wrong += len(groups["tissue_only_correct"])
        pooled_n += n

    print(f"\n{'='*70}\nPOOLED ACROSS ALL 3 FOLDS (n={pooled_n})\n{'='*70}")
    print(f"P(tissue correct | RGB wrong) = {pooled_tissue_correct_given_rgb_wrong}/{pooled_rgb_wrong} = {pooled_tissue_correct_given_rgb_wrong/pooled_rgb_wrong:.4f}")
    print(f"fraction of RGB's errors tissue ALSO misses = {pooled_both_wrong/pooled_rgb_wrong:.4f}")
    print(f"fraction of tissue's errors RGB ALSO misses = {pooled_both_wrong/pooled_tissue_wrong:.4f}")
    oracle_acc = (pooled_n - pooled_both_wrong) / pooled_n
    print(f"[THEORETICAL UPPER BOUND ONLY] pooled oracle selection accuracy = {oracle_acc:.4f}")

    print(f"\n{'='*70}\nFOLD 1 RESIDUAL-FUSION: real vs. shuffled tissue statistics (test set)\n{'='*70}")
    residual_results, n_test = fold1_residual_shuffle_metrics(base_cfg, data_root, feature_lookup, device)
    print(f"n_test={n_test}")
    print(f"{'variant':<18}{'accuracy':>10}{'macro_f1':>10}{'mae':>8}{'qwk':>8}")
    for name in ["rgb_only", "real_fusion", "shuffled_fusion"]:
        m = residual_results[name]
        print(f"{name:<18}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}{m['mae']:>8.4f}{m['qwk']:>8.4f}")
    print("\n(real_fusion should match the already-recorded exp6_results.csv fold-1 row: "
          "accuracy=0.8343 macro_f1=0.8350 qwk=0.9310 -- sanity check, not a new result)")


if __name__ == "__main__":
    main()
