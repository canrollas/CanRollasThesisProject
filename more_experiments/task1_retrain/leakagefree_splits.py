"""Fold construction for the leakage-free Task 1 retrain.

Mirrors task_1/src/splits.py's build_folds (stratified-by-source 3-fold CV,
same seed/val_fraction), but reads the sample list from
../manifests/task1_leakagefree_filelist.txt instead of listing
datasets/wound-region-segmentation/images/ directly -- that raw folder still
contains the 228 excluded PIID images. task_1/src/splits.py itself is left
untouched (original thesis code).
"""
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "task_1"))

from src.splits import infer_source  # noqa: E402

DEFAULT_FILELIST = THIS_DIR.parent / "manifests" / "task1_leakagefree_filelist.txt"


def list_leakagefree_samples(filelist_path=DEFAULT_FILELIST):
    filenames = [line.strip() for line in Path(filelist_path).read_text().splitlines() if line.strip()]
    sources = [infer_source(f) for f in filenames]
    leaked = [f for f, s in zip(filenames, sources) if s == "ft_evre1_stage_1"]
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} PIID Stage-1 image(s) found in the 'leakage-free' filelist "
            f"({filelist_path}) -- regenerate it with scripts/verify_piid_overlap.py "
            f"before training. First offender: {leaked[0]}"
        )
    return filenames, sources


def build_folds(filelist_path=DEFAULT_FILELIST, num_folds=3, val_fraction=0.1, seed=42):
    """Same stratified-by-source 3-fold CV protocol as task_1/src/splits.py,
    applied to the leakage-free sample list.
    """
    filenames, sources = list_leakagefree_samples(filelist_path)
    test_splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    folds = []
    for train_val_idx, test_idx in test_splitter.split(filenames, sources):
        train_val_files = [filenames[i] for i in train_val_idx]
        train_val_sources = [sources[i] for i in train_val_idx]
        test_files = [filenames[i] for i in test_idx]

        val_splits = max(int(round(1 / val_fraction)), 2)
        val_splitter = StratifiedKFold(n_splits=val_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(val_splitter.split(train_val_files, train_val_sources))
        train_files = [train_val_files[i] for i in train_idx]
        val_files = [train_val_files[i] for i in val_idx]

        folds.append({"train": train_files, "val": val_files, "test": test_files})
    return folds
