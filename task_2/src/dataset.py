import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VOID_LABEL = 255

# Fixed class order: index must match configs/base.yaml "classes" and
# src/utils.py RESULT_FIELDS (granulation, slough/fibrin, necrosis).
CLASS_ORDER = ["granulation", "slough", "necrosis"]

# Present in datasets/wound-tissue-segmentation/color-mappings.json but
# excluded from training/evaluation: outside_wound and skin_remnant per the
# paper (too few images for skin_remnant/tendon to learn reliably); bone is
# this project's dataset having a 4th rare category (33/612 images) the
# paper's version didn't include, treated the same way for consistency.
VOID_CLASSES = ["outside_wound", "skin_remnant", "tendon", "bone"]


def _hex_to_rgb(hexcolor):
    hexcolor = hexcolor.lstrip("#")
    return tuple(int(hexcolor[i:i + 2], 16) for i in (0, 2, 4))


def load_color_mapping(data_root):
    # This dataset's color-mappings.json ships with trailing "// Ignore"
    # line comments, which isn't valid JSON — strip them before parsing.
    raw = (Path(data_root) / "color-mappings.json").read_text()
    raw = re.sub(r"//.*", "", raw)
    mapping = json.loads(raw)
    class_colors = [_hex_to_rgb(mapping[name]) for name in CLASS_ORDER]
    return class_colors


def mask_rgb_to_index(mask_rgb, class_colors):
    """Unmatched colors (including all VOID_CLASSES) default to VOID_LABEL."""
    index = np.full(mask_rgb.shape[:2], VOID_LABEL, dtype=np.int64)
    for class_id, rgb in enumerate(class_colors):
        matches = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        index[matches] = class_id
    return index


def image_to_mask_filename(image_filename):
    return image_filename.replace("_wound_", "_mask_")


class TissueDataset(Dataset):
    def __init__(self, filenames, data_root, image_size, class_colors, transform=None):
        self.filenames = filenames
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.class_colors = class_colors
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        mask_filename = image_to_mask_filename(filename)
        # RGBA alpha channel marks the wound region, but outside_wound pixels
        # are also annotated black in the RGB mask, so dropping alpha here
        # and resolving void purely from mask color (mask_rgb_to_index) is
        # equivalent and keeps the pipeline 3-channel like task_1's.
        image = Image.open(self.data_root / "tissue_images" / filename).convert("RGB")
        mask = Image.open(self.data_root / "tissue_masks" / mask_filename).convert("RGB")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_np = np.array(image, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)

        if self.transform is not None:
            image_np, mask_np = self.transform(image_np, mask_np)

        label = mask_rgb_to_index(mask_np, self.class_colors)

        image_f = image_np.astype(np.float32) / 255.0
        image_f = (image_f - IMAGENET_MEAN) / IMAGENET_STD
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).float()
        label_t = torch.from_numpy(label).long()

        return image_t, label_t
