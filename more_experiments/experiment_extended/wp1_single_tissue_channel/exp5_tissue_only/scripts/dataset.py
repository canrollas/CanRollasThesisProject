"""Work Package 1 (advisor extended experiments): combined-tissue-channel
two-stream dataset for the exp-5 (residual fusion, tissue_only) architecture.

Per the advisor's own note ("Tek kanal exp-4 ve exp-5 denenecek" -- single
channel exp-4/exp-5 will be tried, i.e. 5 kanal yerine tum dokular tek
kanalda verilsin / all five tissues given in a single channel instead of
5): unlike exp-5's own dataset.py (which concatenates all 5 tissue
probability channels), this collapses all 5 into exactly ONE channel via a
per-pixel maximum (max(granulation, slough, necrosis, tendon, bone) at
every pixel) before handing it to the exact same residual-fusion
architecture. Tissue-type identity is deliberately discarded; only "how
much evidence of ANY non-healthy tissue type is here" survives, so this
directly tests whether exp-5's 5-channel fusion needs per-type
distinction or whether a single combined magnitude is enough.

Reuses the EXACT SAME .npy files exp-3/exp-4/exp-5 already generated
(datasets/wound-staging-masks-exp-3/, 8-channel: R,G,B,granulation,slough,
necrosis,tendon,bone -- see that dataset's provenance.json) -- no new
dataset generation, the channel merge happens on load.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Tissue channels occupy indices 3-7 of the 8-channel array (0,1,2 = R,G,B);
# combined into one channel via a per-pixel max (see module docstring).
TISSUE_CHANNEL_SLICE = slice(3, 8)


def _resize_rgb_and_combined_tissue(arr, size):
    rgb = arr[:, :, :3]
    tissue = arr[:, :, TISSUE_CHANNEL_SLICE].max(axis=-1)  # (H, W), per-pixel max over the 5 tissue types
    if arr.shape[0] == size and arr.shape[1] == size:
        return rgb, tissue
    rgb_r = np.array(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))
    tissue_r = np.array(Image.fromarray(tissue).resize((size, size), Image.BILINEAR))
    return rgb_r, tissue_r


class TwoStreamCombinedTissueDataset(Dataset):
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
        rgb, tissue = _resize_rgb_and_combined_tissue(arr, self.image_size)

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
