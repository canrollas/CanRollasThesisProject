import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Fixed class order: index must match src/metrics.py and configs/base.yaml
# "classes" list (background, skin, wound).
CLASS_ORDER = ["other", "skin", "wound"]


def load_color_mapping(data_root):
    with open(Path(data_root) / "color-mappings.json") as f:
        mapping = json.load(f)
    colors = []
    for name in CLASS_ORDER:
        hexcolor = mapping[name].lstrip("#")
        rgb = tuple(int(hexcolor[i:i + 2], 16) for i in (0, 2, 4))
        colors.append(rgb)
    return colors


def mask_rgb_to_index(mask_rgb, colors):
    index = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    for class_id, rgb in enumerate(colors):
        matches = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        index[matches] = class_id
    return index


class WoundRegionDataset(Dataset):
    def __init__(self, filenames, data_root, image_size, colors, transform=None):
        self.filenames = filenames
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.colors = colors
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(self.data_root / "images" / filename).convert("RGB")
        mask = Image.open(self.data_root / "masks" / filename).convert("RGB")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_np = np.array(image, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)

        if self.transform is not None:
            image_np, mask_np = self.transform(image_np, mask_np)

        label = mask_rgb_to_index(mask_np, self.colors)

        image_f = image_np.astype(np.float32) / 255.0
        image_f = (image_f - IMAGENET_MEAN) / IMAGENET_STD
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).float()
        label_t = torch.from_numpy(label).long()

        return image_t, label_t
