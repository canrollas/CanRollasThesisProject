"""Dataset for exp-7: (rgb_tensor, box_tensor[1,4], label). RGB preprocessing
is IDENTICAL to task_3/src/dataset.py's StageDataset (same resize, same
ImageNet mean/std) -- the only difference is also returning the ROI box
looked up from ../results/roi_manifest.csv by filename stem, in the SAME
224x224 coordinate system the RGB resize produces.
"""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_roi_manifest(csv_path):
    lookup = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            lookup[row["filename"]] = {
                "box": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                "detected": row["detected"] == "True",
                "wound_area_pct": float(row["wound_area_pct"]),
                "fold": int(row["fold"]),
            }
    return lookup


class ROIStageDataset(Dataset):
    def __init__(self, samples, data_root, roi_lookup, image_size=224, augment=None, box_override=None):
        """samples: list of (rel_path, label) as returned by
        task_3/src/splits.py build_folds. augment: an
        augmentations.PairedImageBoxAugmentation instance (train split
        only -- keeps the box geometrically in sync with the flipped/
        rotated image, see augmentations.py) or None (val/test). box_override:
        optional dict {stem: [x1,y1,x2,y2]} to substitute the manifest's box
        (used by the inference-only ROI ablation -- full-image/shuffled
        variants); augmentation is never applied together with an override
        (those ablations are eval-only)."""
        self.samples = samples
        self.data_root = Path(data_root)
        self.roi_lookup = roi_lookup
        self.image_size = image_size
        self.augment = augment
        self.box_override = box_override

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        stem = Path(rel_path).stem

        image = Image.open(self.data_root / rel_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.array(image, dtype=np.uint8)

        box = self.box_override[stem] if self.box_override is not None else self.roi_lookup[stem]["box"]

        if self.augment is not None:
            assert self.box_override is None, "augmentation + box_override should never be combined"
            image_np, box = self.augment(image_np, box)

        image_f = image_np.astype(np.float32) / 255.0
        image_f = (image_f - IMAGENET_MEAN) / IMAGENET_STD
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).float()
        box_t = torch.tensor([box], dtype=torch.float32)  # [1, 4]

        return image_t, box_t, label, stem
