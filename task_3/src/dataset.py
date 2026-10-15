from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Fixed class order: index must match configs/base.yaml "classes" and the
# datasets/wound-stage-classification/ directory names (one dir per stage).
CLASS_ORDER = ["stage_1", "stage_2", "stage_3", "stage_4"]


def list_samples(data_root):
    """Returns (relative_path, class_idx) pairs across all stage_*/
    subdirectories, sorted per class for reproducibility."""
    data_root = Path(data_root)
    samples = []
    for class_idx, class_dir in enumerate(CLASS_ORDER):
        for p in sorted((data_root / class_dir).iterdir()):
            if p.is_file():
                samples.append((f"{class_dir}/{p.name}", class_idx))
    return samples


class StageDataset(Dataset):
    def __init__(self, samples, data_root, image_size, transform=None):
        self.samples = samples
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        image = Image.open(self.data_root / rel_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.array(image, dtype=np.uint8)

        if self.transform is not None:
            image_np = self.transform(image_np)

        image_f = image_np.astype(np.float32) / 255.0
        image_f = (image_f - IMAGENET_MEAN) / IMAGENET_STD
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).float()

        return image_t, label
