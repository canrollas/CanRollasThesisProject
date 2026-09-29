import re
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

# Ordered so more specific prefixes are checked before shorter ones that
# could otherwise shadow them (none currently overlap, but keep this order
# if new sources are added).
SOURCE_PATTERNS = [
    ("ft_evre1_stage_1", re.compile(r"^ft_evre1_stage_1_")),
    ("ft_tendonbone", re.compile(r"^ft_tendonbone_")),
    ("fusc", re.compile(r"^fusc_")),
    ("wsnet", re.compile(r"^wsnet_")),
    ("medetec", re.compile(r"^medetec_")),
]


def infer_source(filename):
    for name, pattern in SOURCE_PATTERNS:
        if pattern.match(filename):
            return name
    return "other"


def list_samples(data_root):
    images_dir = Path(data_root) / "images"
    filenames = sorted(p.name for p in images_dir.iterdir() if p.is_file())
    sources = [infer_source(f) for f in filenames]
    return filenames, sources


def build_folds(data_root, num_folds=3, val_fraction=0.1, seed=42):
    """3-fold CV stratified by source, each fold's test chunk held out in
    turn; val_fraction of the remaining train_val pool is carved out
    (also stratified by source) for checkpoint selection.
    """
    filenames, sources = list_samples(data_root)
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
