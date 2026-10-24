import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import image_to_mask_filename, mask_rgb_to_index


def list_samples(data_root):
    images_dir = Path(data_root) / "tissue_images"
    return sorted(p.name for p in images_dir.iterdir() if p.is_file())


def _stratum_seed(seed, key):
    """Deterministic per-stratum seed, stable across processes and Python
    versions. A plain `seed ^ hash(str(key))` broke under PYTHONHASHSEED
    randomization: training and evaluation run as separate processes, so
    each one silently drew a different split from the same `seed`. Measured
    on this exact dataset before the fix: 51 of 69 test images in one run
    (74%) were in another run's training set. md5 is stable across
    processes, which the built-in str hash is not.
    """
    digest = hashlib.md5(str(key).encode()).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def _presence_stratum(filename, data_root, class_colors):
    """Stratum key = which classes are present in a downsampled mask,
    rarest class first (bone, tendon, necrosis, slough, granulation), so a
    k-fold split balanced on this key keeps the rarest classes represented
    across folds instead of landing unevenly by chance.
    """
    mask_path = Path(data_root) / "tissue_masks" / image_to_mask_filename(filename)
    mask = Image.open(mask_path).convert("RGB").resize((64, 64), Image.NEAREST)
    label = mask_rgb_to_index(np.array(mask, dtype=np.uint8), class_colors)
    present = set(int(c) for c in np.unique(label) if c != 255)
    return tuple(int(c in present) for c in reversed(range(len(class_colors))))


def build_folds(data_root, class_colors, num_folds=3, val_fraction=0.1, seed=42):
    """Stratified k-fold split, ported from the segmentation-benchmarking
    project's utils.dataset._stratified_kfold_split. Images are grouped into
    strata by _presence_stratum, each stratum is shuffled once with a stable
    per-stratum seed and cut into num_folds balanced chunks (remainder goes
    to the first chunks); each fold holds out one chunk per stratum as test,
    then carves val_fraction of the rest (reshuffled with the plain `seed`)
    for checkpoint selection.
    """
    filenames = list_samples(data_root)
    strata = [_presence_stratum(f, data_root, class_colors) for f in filenames]

    groups = defaultdict(list)
    for f, s in zip(filenames, strata):
        groups[s].append(f)

    test_chunks = [[] for _ in range(num_folds)]
    other_chunks = [[] for _ in range(num_folds)]
    for key in sorted(groups):
        grp = list(groups[key])
        random.Random(_stratum_seed(seed, key)).shuffle(grp)
        n = len(grp)
        fold_sizes = [n // num_folds + (1 if i < n % num_folds else 0) for i in range(num_folds)]
        boundaries = [0]
        for fs in fold_sizes:
            boundaries.append(boundaries[-1] + fs)
        for i in range(num_folds):
            test_chunks[i].extend(grp[boundaries[i]:boundaries[i + 1]])
            other_chunks[i].extend(grp[:boundaries[i]] + grp[boundaries[i + 1]:])

    folds = []
    for i in range(num_folds):
        other_files = other_chunks[i][:]
        random.Random(seed).shuffle(other_files)
        n_val = max(1, int(len(other_files) * val_fraction))
        folds.append({
            "train": sorted(other_files[n_val:]),
            "val": sorted(other_files[:n_val]),
            "test": sorted(test_chunks[i]),
        })
    return folds
