import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def list_filenames(images_dir):
    return sorted(p.name for p in Path(images_dir).glob("*.png"))


def split_holdout(filenames, holdout_fraction, seed):
    """Splits off a fixed test set that is carved out ONCE and never touched
    by any ablation run. With ~40+ configs being compared per axis, picking
    a "best" value using the same rotating k-fold test folds that get
    reported would bias the final numbers; this holdout set gives the final
    recommended config a clean, never-selected-on evaluation. The k-fold CV
    used for every ablation axis operates only on the remaining dev pool."""
    shuffled = list(filenames)
    random.Random(seed).shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * holdout_fraction))
    holdout = sorted(shuffled[:n_holdout])
    dev_pool = sorted(shuffled[n_holdout:])
    return dev_pool, holdout


def make_kfold_splits(filenames, n_folds, seed, val_fraction_of_remaining=0.1):
    """K-fold protocol matching the source paper: shuffle once with a fixed
    seed, cut into n_folds consecutive chunks, and for each fold use its
    chunk as the test set and carve a validation set out of the remainder.
    Returns a list of (train, val, test) filename lists, one per fold."""
    shuffled = list(filenames)
    random.Random(seed).shuffle(shuffled)

    fold_size = len(shuffled) // n_folds
    folds = []
    start = 0
    for i in range(n_folds):
        end = start + fold_size if i < n_folds - 1 else len(shuffled)
        folds.append(shuffled[start:end])
        start = end

    splits = []
    for i in range(n_folds):
        test = folds[i]
        remaining = [f for j, chunk in enumerate(folds) if j != i for f in chunk]
        rng = random.Random(seed + i)
        remaining = remaining[:]
        rng.shuffle(remaining)
        n_val = max(1, int(len(remaining) * val_fraction_of_remaining))
        val, train = remaining[:n_val], remaining[n_val:]
        splits.append((train, val, test))
    return splits


def subsample(filenames, fraction, seed):
    if fraction >= 1.0:
        return filenames
    rng = random.Random(seed)
    n = max(1, int(len(filenames) * fraction))
    return rng.sample(filenames, n)


class WoundRegionDataset(Dataset):
    """RGB wound photographs with RGB color-coded segmentation masks
    (classes: other / skin / wound). Images and masks share the same
    filename in their respective directories."""

    def __init__(self, images_dir, masks_dir, filenames, classes, color_mapping, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.filenames = filenames
        self.classes = classes
        self.color_mapping = color_mapping
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def _mask_to_label(self, mask_rgb):
        label = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        for idx, cls in enumerate(self.classes):
            color = np.array(self.color_mapping[cls], dtype=np.uint8)
            matches = np.all(mask_rgb == color, axis=-1)
            label[matches] = idx
        return label

    def __getitem__(self, index):
        fname = self.filenames[index]
        image = np.array(Image.open(self.images_dir / fname).convert("RGB"))
        mask_rgb = np.array(Image.open(self.masks_dir / fname).convert("RGB"))
        label = self._mask_to_label(mask_rgb)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented["image"], augmented["mask"]

        if not torch.is_tensor(image):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if not torch.is_tensor(label):
            label = torch.from_numpy(label)
        label = label.long()

        return image, label
