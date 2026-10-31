"""4-channel dataset loader for exp-14 (RGB + 1 monocular relative-depth
channel).

Mechanically identical to exp-3.1's scripts/dataset.py (same .npy layout:
(H, W, 4) uint8, RGB + 1 extra channel) -- only the docstring/name differ,
since the loader itself doesn't care what the 4th channel represents
(tissue severity there, relative depth here), only that it's a plain
[0, 255] scale, not an ImageNet-normalized RGB channel.

list_samples/build_folds are NOT reimplemented here -- task_3/src/splits.py's
own versions work unchanged on this dataset (verified identically to
exp-3/exp-3.1: fold membership only depends on stage_X/ directory names and
sorted filenames, and the .npy extension doesn't change sort order relative
to the original .jpg files), so fold membership stays identical to task_3's
own 3-fold split.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_multichannel(arr, size):
    """PIL can't load >4 channels at once in general, so RGB and the depth
    channel are resized individually and restacked. No-op when already the
    right size (true for every current exp-14 sample, which are all
    224x224, matching task_3's and exp-3.1's own image_size)."""
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    rgb = np.array(Image.fromarray(arr[:, :, :3]).resize((size, size), Image.BILINEAR))
    depth = np.array(Image.fromarray(arr[:, :, 3]).resize((size, size), Image.BILINEAR))
    return np.concatenate([rgb, depth[:, :, None]], axis=-1)


class DepthChannelStageDataset(Dataset):
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
        depth = arr[:, :, 3:].astype(np.float32) / 255.0

        combined = np.concatenate([rgb, depth], axis=-1)
        image_t = torch.from_numpy(combined.transpose(2, 0, 1)).float()
        return image_t, label
