import random

from .dataset import CLASS_ORDER, list_samples


def build_folds(data_root, num_folds=3, val_fraction=0.1, seed=42):
    """Stratified version of task_1/task_3's shuffle-then-chunk protocol:
    each class is shuffled and chunked independently (fixed per-class seed),
    then per-class chunks are concatenated into num_folds overall chunks.
    Keeps each fold's class balance close to the full dataset's (230/313/
    275/273 for stage_1..stage_4) instead of letting one global shuffle
    stack a stage disproportionately into a single fold.
    """
    samples = list_samples(data_root)
    by_class = {c: [] for c in range(len(CLASS_ORDER))}
    for sample in samples:
        by_class[sample[1]].append(sample)

    chunks = [[] for _ in range(num_folds)]
    for class_idx, class_samples in by_class.items():
        rng = random.Random(seed + class_idx)
        shuffled = class_samples[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        chunk_size = n // num_folds
        start = 0
        for i in range(num_folds):
            end = start + chunk_size if i < num_folds - 1 else n
            chunks[i].extend(shuffled[start:end])
            start = end

    folds = []
    for i in range(num_folds):
        test_samples = chunks[i]
        train_val_samples = [s for j, chunk in enumerate(chunks) if j != i for s in chunk]

        tv_rng = random.Random(seed + 100 + i)
        tv_shuffled = train_val_samples[:]
        tv_rng.shuffle(tv_shuffled)
        n_val = max(int(round(len(tv_shuffled) * val_fraction)), 1)

        folds.append({
            "train": tv_shuffled[n_val:],
            "val": tv_shuffled[:n_val],
            "test": test_samples,
        })
    return folds
