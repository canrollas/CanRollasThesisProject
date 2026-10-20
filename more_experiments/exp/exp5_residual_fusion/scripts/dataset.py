"""Two-stream dataset for exp-5: RGB and tissue channels returned as
SEPARATE tensors, from the same datasets/wound-staging-masks-exp-3/ .npy
files exp-3/exp-4 use -- same pixels, same leakage-safe Task 2 tissue
predictions, same verified PIID fold split.

Intentionally a direct copy of exp-4's TwoStreamStageDataset (not a
cross-module import): a dynamically-loaded module (importlib trick) can't
be pickled by PyTorch DataLoader's multiprocessing workers when
--workers > 0 ("No module named 'exp4_late_fusion_dataset'" in a spawned
worker process), so this stays a small, independently-importable
duplicate rather than trading reuse for broken multi-worker loading.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_TISSUE_CHANNELS = 5


def _resize_multichannel(arr, size):
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    rgb = np.array(Image.fromarray(arr[:, :, :3]).resize((size, size), Image.BILINEAR))
    tissue_channels = [
        np.array(Image.fromarray(arr[:, :, 3 + i]).resize((size, size), Image.BILINEAR))
        for i in range(NUM_TISSUE_CHANNELS)
    ]
    return np.concatenate([rgb, np.stack(tissue_channels, axis=-1)], axis=-1)


class TwoStreamStageDataset(Dataset):
    def __init__(self, samples, data_root, image_size, transform=None):
        self.samples = samples
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        arr = np.load(self.data_root / rel_path)  # (H, W, 8) uint8
        arr = _resize_multichannel(arr, self.image_size)

        if self.transform is not None:
            arr = self.transform(arr)  # flip/rotate applied jointly, before the split

        rgb = arr[:, :, :3].astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float()

        tissue = arr[:, :, 3:].astype(np.float32) / 255.0
        tissue_t = torch.from_numpy(tissue.transpose(2, 0, 1)).float()

        return rgb_t, tissue_t, label
