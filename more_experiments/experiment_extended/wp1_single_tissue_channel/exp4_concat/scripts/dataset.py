"""Work Package 1 (advisor extended experiments): single-tissue-channel
two-stream dataset for the exp-4 (late fusion, concat) architecture.

Reuses the EXACT SAME .npy files exp-3/exp-4/exp-5 already generated
(datasets/wound-staging-masks-exp-3/, 8-channel: R,G,B,granulation,slough,
necrosis,tendon,bone -- see that dataset's provenance.json) -- no new
dataset generation. Unlike exp-4's own dataset.py (which returns all 5
tissue channels), this slices out exactly ONE of the 5 channels, selected
by name at construction time, so --tissue-channel sweeps granulation/
slough/necrosis/tendon/bone independently without duplicating any data on
disk.

Channel selection is passed into the Dataset object's __init__ (plain data
on the instance, not a dynamically-imported sibling module), so this stays
safe under DataLoader multiprocessing with --workers > 0, same as every
other dataset.py in this repo.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Absolute channel index within the 8-channel array (0,1,2 = R,G,B).
TISSUE_CHANNEL_INDEX = {"granulation": 3, "slough": 4, "necrosis": 5, "tendon": 6, "bone": 7}


def _resize_rgb_and_one_channel(arr, size, channel_idx):
    rgb = arr[:, :, :3]
    tissue = arr[:, :, channel_idx]
    if arr.shape[0] == size and arr.shape[1] == size:
        return rgb, tissue
    rgb_r = np.array(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))
    tissue_r = np.array(Image.fromarray(tissue).resize((size, size), Image.BILINEAR))
    return rgb_r, tissue_r


class TwoStreamSingleChannelDataset(Dataset):
    def __init__(self, samples, data_root, image_size, tissue_channel, transform=None):
        if tissue_channel not in TISSUE_CHANNEL_INDEX:
            raise ValueError(f"tissue_channel must be one of {list(TISSUE_CHANNEL_INDEX)}, got {tissue_channel!r}")
        self.samples = samples
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.channel_idx = TISSUE_CHANNEL_INDEX[tissue_channel]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        arr = np.load(self.data_root / rel_path)  # (H, W, 8) uint8
        rgb, tissue = _resize_rgb_and_one_channel(arr, self.image_size, self.channel_idx)

        if self.transform is not None:
            combined = np.concatenate([rgb, tissue[:, :, None]], axis=-1)  # flip/rotate jointly, before the split
            combined = self.transform(combined)
            rgb, tissue = combined[:, :, :3], combined[:, :, 3]

        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_f = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD
        rgb_t = torch.from_numpy(rgb_f.transpose(2, 0, 1)).float()

        tissue_f = tissue.astype(np.float32) / 255.0
        tissue_t = torch.from_numpy(tissue_f).float().unsqueeze(0)  # [1, H, W]

        return rgb_t, tissue_t, label
