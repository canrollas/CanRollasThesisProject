"""8-channel dataset loader for exp-3 (RGB + 5 tissue-probability channels).

Mirrors task_3/src/dataset.py's StageDataset, but loads the .npy arrays
generate_dataset.py writes (H x W x 8 uint8) instead of JPEGs, and
normalizes the two channel groups differently: RGB with ImageNet
mean/std (matching the pretrained backbone's expected input stats),
tissue channels with a plain /255 scale (they're already [0,1]
probabilities stored as uint8, not natural-image RGB -- ImageNet stats
would be meaningless for them).

list_samples/build_folds are NOT reimplemented here -- task_3/src/splits.py's
own versions work unchanged on this dataset (they only look at stage_X/
directory names and sort filenames; the .npy extension doesn't affect
sort order relative to the original .jpg files, verified directly before
relying on it), so fold membership stays identical to task_3's own split.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_TISSUE_CHANNELS = 5  # granulation, slough, necrosis, tendon, bone


def _resize_multichannel(arr, size):
    """PIL can't load >4 channels at once, so RGB and each tissue channel
    are resized individually and restacked. No-op when already the right
    size (true for every current exp-3 sample, which are all 224x224)."""
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    rgb = np.array(Image.fromarray(arr[:, :, :3]).resize((size, size), Image.BILINEAR))
    tissue_channels = [
        np.array(Image.fromarray(arr[:, :, 3 + i]).resize((size, size), Image.BILINEAR))
        for i in range(NUM_TISSUE_CHANNELS)
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
        arr = np.load(self.data_root / rel_path)  # (H, W, 8) uint8
        arr = _resize_multichannel(arr, self.image_size)

        if self.transform is not None:
            arr = self.transform(arr)  # flip/rotate: channel-count-agnostic, works as-is

        rgb = arr[:, :, :3].astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tissue = arr[:, :, 3:].astype(np.float32) / 255.0

        combined = np.concatenate([rgb, tissue], axis=-1)
        image_t = torch.from_numpy(combined.transpose(2, 0, 1)).float()
        return image_t, label
