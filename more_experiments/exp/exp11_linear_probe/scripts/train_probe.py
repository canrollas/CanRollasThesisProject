#!/usr/bin/env python3
"""Experiment 11 (Chapter 8): linear-probe diagnostic for CORN's errors
-- Step 3, probe training.

`sklearn.linear_model.LogisticRegression` (multinomial, L2) on top of
the frozen features `extract_features.py` saved -- the "cleaner" linear
probe the proposal itself suggests over a hand-rolled `torch.nn.Linear`,
since scikit-learn's L-BFGS solver converges to the actual optimum
instead of depending on an arbitrary number of SGD epochs. Features are
standardized with a `StandardScaler` FIT ONLY ON TRAIN (val/test are
`.transform()`-ed with train's fitted mean/std, never refit) -- the
regularization strength `C` is grid-searched by VALIDATION macro-F1
only; the test split is touched exactly once, after `C` is already
fixed, matching the proposal's explicit "no training or hyperparameter
selection may touch test" requirement.

Run once per (fold, feature level) -- `--level final` (CornHead's own
512-dim input) and `--level stage3` (the earlier, 320-dim control level)
are two separate linear probes, not a shared one; see
extract_features.py's docstring for why "stage3" substitutes for a
"pre-logit CORN feature" that does not exist in this architecture.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.metrics import compute_metrics  # noqa: E402

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def load_split(fold_dir, split, level):
    features = np.load(fold_dir / f"{split}_{level}_features.npy")
    labels = np.load(fold_dir / f"{split}_labels.npy")
    return features, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--level", choices=("final", "stage3"), required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fold_dir = EXP_DIR / "results" / f"fold_{args.fold}"
    if not fold_dir.is_dir():
        raise SystemExit(f"BLOCKED: {fold_dir} not found -- run extract_features.py --fold {args.fold} first.")

    x_train, y_train = load_split(fold_dir, "train", args.level)
    x_val, y_val = load_split(fold_dir, "val", args.level)
    x_test, y_test = load_split(fold_dir, "test", args.level)
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)

    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    best_c, best_val_f1, best_clf = None, -1.0, None
    val_scores = {}
    for c in C_GRID:
        # lbfgs (the default solver) already fits a multinomial model for a
        # >2-class target on its own -- sklearn 1.9 removed the explicit
        # `multi_class` kwarg entirely (used to default to "auto" in
        # older versions), so it's not passed here.
        clf = LogisticRegression(C=c, max_iter=2000, random_state=args.seed)
        clf.fit(x_train_s, y_train)
        val_pred = clf.predict(x_val_s)
        val_f1 = f1_score(y_val, val_pred, average="macro", labels=list(range(num_classes)), zero_division=0)
        val_scores[c] = round(float(val_f1), 4)
        if val_f1 > best_val_f1:
            best_val_f1, best_c, best_clf = val_f1, c, clf

    train_pred = best_clf.predict(x_train_s)
    val_pred = best_clf.predict(x_val_s)
    test_pred = best_clf.predict(x_test_s)  # touched exactly once, after C is already fixed on val
    # Class probabilities too -- not used for anything in THIS script, but
    # needed downstream (oracle_analysis.py's disagreement-confidence
    # check) without having to refit the probe there.
    test_proba = best_clf.predict_proba(x_test_s)

    train_stats = compute_metrics(y_train, train_pred, num_classes)
    val_stats = compute_metrics(y_val, val_pred, num_classes)
    test_stats = compute_metrics(y_test, test_pred, num_classes)

    for stats in (train_stats, val_stats, test_stats):
        stats.pop("confusion")  # ndarray, not JSON-serializable -- compare_errors.py recomputes
                                  # any confusion/stage-pair breakdown it needs from the saved preds directly

    np.save(fold_dir / f"probe_{args.level}_train_preds.npy", train_pred)
    np.save(fold_dir / f"probe_{args.level}_val_preds.npy", val_pred)
    np.save(fold_dir / f"probe_{args.level}_test_preds.npy", test_pred)
    np.save(fold_dir / f"probe_{args.level}_test_proba.npy", test_proba)
    np.save(fold_dir / f"probe_{args.level}_classes.npy", best_clf.classes_)  # column order for test_proba

    summary = {
        "fold": args.fold, "level": args.level, "feature_dim": int(x_train.shape[1]),
        "c_grid_val_macro_f1": val_scores, "selected_c": best_c,
        "train": train_stats, "val": val_stats, "test": test_stats,
    }
    with open(fold_dir / f"probe_{args.level}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[fold{args.fold} {args.level}] selected C={best_c} (val_macro_f1={best_val_f1:.4f})")
    print(f"[fold{args.fold} {args.level}] test: acc={test_stats['accuracy']:.4f} "
          f"macro_f1={test_stats['macro_f1']:.4f} qwk={test_stats['qwk']:.4f}")
    print(f"wrote {fold_dir / f'probe_{args.level}_summary.json'}")


if __name__ == "__main__":
    main()
