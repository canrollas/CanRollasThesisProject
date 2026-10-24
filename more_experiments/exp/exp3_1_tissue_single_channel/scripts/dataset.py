"""4-channel dataset loader for exp-3.1 (RGB + 1 severity-banded tissue channel).

Mirrors exp-3's scripts/dataset.py, but generalized to any tissue-channel
count instead of hardcoding 5 -- this experiment's .npy arrays are
(H x W x 4) rather than (H x W x 8). Normalizes the two channel groups
differently: RGB with ImageNet mean/std (matching the pretrained
backbone's expected input stats), the tissue channel with a plain /255
scale (it's a [0, 255] severity-band encoding stored as uint8, not a
natural-image RGB channel -- ImageNet stats would be meaningless for it).

list_samples/build_folds are NOT reimplemented here -- task_3/src/splits.py's
own versions work unchanged on this dataset (they only look at stage_X/
directory names and sort filenames; the .npy extension doesn't affect sort
order relative to the original .jpg files, verified directly in exp-3
before relying on it, same reasoning applies here), so fold membership
stays identical to task_3's and exp-3's own split.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_multichannel(arr, size):
    """PIL can't load >4 channels at once in general, so RGB and each
    tissue channel are resized individually and restacked. Channel count
    agnostic (unlike exp-3's version, which hardcoded 5 tissue channels).
    No-op when already the right size (true for every current exp-3.1
    sample, which are all 224x224)."""
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    num_tissue_channels = arr.shape[-1] - 3
    rgb = np.array(Image.fromarray(arr[:, :, :3]).resize((size, size), Image.BILINEAR))
    tissue_channels = [
        np.array(Image.fromarray(arr[:, :, 3 + i]).resize((size, size), Image.BILINEAR))
        for i in range(num_tissue_channels)
    ]
    return np.concatenate([rgb, np.stack(tissue_channels, axis=-1)], axis=-1)


class TissueChannelStageDataset(Dataset):
    def __init__(self, samples, data_root, image_size, transform=None):
        self.samples = samples
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        arr = np.load(self.data_root / rel_path)  # (H, W, 4) uint8
        arr = _resize_multichannel(arr, self.image_size)

        if self.transform is not None:
            arr = self.transform(arr)  # flip/rotate: channel-count-agnostic, works as-is

        rgb = arr[:, :, :3].astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tissue = arr[:, :, 3:].astype(np.float32) / 255.0

        combined = np.concatenate([rgb, tissue], axis=-1)
        image_t = torch.from_numpy(combined.transpose(2, 0, 1)).float()
        return image_t, label
