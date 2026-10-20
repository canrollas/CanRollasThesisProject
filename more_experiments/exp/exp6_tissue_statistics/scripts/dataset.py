"""Datasets for exp-6 (hand-crafted tissue statistics, see
extract_tissue_features.py). Two variants:

  TissueStatsOnlyDataset: (stats_vector, label) -- no RGB at all, matches
  task_3/src/engine.py's model(images)->logits signature exactly (the
  "images" here are just the 20-dim stats vector), so that engine is
  reused UNCHANGED for the tissue_statistics_only experiment.

  TissueStatsResidualDataset: (rgb_tensor, stats_vector, label) -- matches
  exp5/scripts/engine.py's model(rgb, tissue)->(z_final,z_rgb,delta_z)
  signature exactly, so THAT engine is reused unchanged too for
  tissue_statistics_residual_fusion.

Both read the RGB (when needed) from datasets/wound-staging-masks-exp-3/'s
.npy files (ignoring their tissue channels -- those are superseded by the
precomputed statistics) and look up the 20-dim vector from
../tissue_statistics.csv by filename stem.
"""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FEATURE_COLUMNS = [
    f"{c}_{s}" for c in ["granulation", "slough", "necrosis", "tendon", "bone"]
    for s in ["mean", "std", "p95", "frac_above_0.5"]
]


def load_feature_table(csv_path):
    lookup = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            lookup[row["filename"]] = np.array([float(row[c]) for c in FEATURE_COLUMNS], dtype=np.float32)
    return lookup


def _stats_for(rel_path, feature_lookup):
    stem = Path(rel_path).stem  # "stage_1/stage_1_10267.npy" -> "stage_1_10267"
    return torch.from_numpy(feature_lookup[stem]).float()


class TissueStatsOnlyDataset(Dataset):
    def __init__(self, samples, feature_lookup):
        self.samples = samples
        self.feature_lookup = feature_lookup

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        return _stats_for(rel_path, self.feature_lookup), label


class TissueStatsResidualDataset(Dataset):
    def __init__(self, samples, data_root, feature_lookup, image_size, transform=None):
        self.samples = samples
        self.data_root = Path(data_root)
        self.feature_lookup = feature_lookup
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        arr = np.load(self.data_root / rel_path)  # (H, W, 8) uint8
        rgb = arr[:, :, :3]

        if self.transform is not None:
            rgb = self.transform(rgb)

        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_f = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD
        rgb_t = torch.from_numpy(rgb_f.transpose(2, 0, 1)).float()

        return rgb_t, _stats_for(rel_path, self.feature_lookup), label
