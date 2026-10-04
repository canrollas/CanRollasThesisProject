import random
from pathlib import Path


def list_samples(data_root):
    images_dir = Path(data_root) / "tissue_images"
    return sorted(p.name for p in images_dir.iterdir() if p.is_file())


def build_folds(data_root, num_folds=3, val_fraction=0.1, seed=42):
    """Matches the paper's Sec. 4.4 protocol: shuffle once with a fixed seed,
    then split into num_folds consecutive chunks (the last chunk absorbs any
    remainder). Each fold holds out one chunk as its test set; val_fraction
    of the remaining pool (reshuffled per fold) is carved out for checkpoint
    selection, the rest is train.
    """
    filenames = list_samples(data_root)
    rng = random.Random(seed)
    shuffled = filenames[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    chunk_size = n // num_folds
    chunks = []
    start = 0
    for i in range(num_folds):
        end = start + chunk_size if i < num_folds - 1 else n
        chunks.append(shuffled[start:end])
        start = end

    folds = []
    for i in range(num_folds):
        test_files = chunks[i]
        train_val_files = [f for j, chunk in enumerate(chunks) if j != i for f in chunk]

        tv_rng = random.Random(seed + i)
        tv_shuffled = train_val_files[:]
        tv_rng.shuffle(tv_shuffled)
        n_val = max(int(round(len(tv_shuffled) * val_fraction)), 1)

        folds.append({
            "train": tv_shuffled[n_val:],
            "val": tv_shuffled[:n_val],
            "test": test_files,
        })
    return folds
