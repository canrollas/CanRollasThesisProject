"""Aligned original RGB, cached tissue probabilities, and optional relative depth.

Depth is an image-derived relative inverse-depth prior, never measured wound
depth. Missing depth is explicit; corrupt existing maps always raise an error.
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_float(array, size):
    return np.asarray(
        Image.fromarray(array.astype(np.float32)).resize(
            (size, size), Image.Resampling.BILINEAR
        ), dtype=np.float32,
    )


def _joint_augment(array):
    # Same draw order and probabilities as task_3/src/augmentations.py.
    if random.random() < 0.5:
        array = array[:, ::-1]
    if random.random() < 0.5:
        array = array[::-1, :]
    if random.random() < 0.7:
        array = np.rot90(array, random.choice((90, 180, 270)) // 90)
    return np.ascontiguousarray(array)


def validate_depth(depth, expected_shape, path):
    """Reject quantized visualizations, mismatched maps, NaNs and flat outputs."""
    if depth.dtype != np.float32 or depth.shape != expected_shape:
        raise ValueError(
            f"Depth must be float32 with original image shape {expected_shape}: "
            f"{path} has {depth.dtype} {depth.shape}"
        )
    if not np.isfinite(depth).all():
        raise ValueError(f"Non-finite relative-depth values: {path}")
    if float(np.ptp(depth)) <= 1e-6:
        raise ValueError(f"Flat relative-depth output: {path}")


class StagePriorDataset(Dataset):
    def __init__(self, samples, data_root, tissue_root, depth_root,
                 image_size=224, augment=False, require_depth=False):
        self.samples = list(samples)
        self.data_root = Path(data_root)
        self.tissue_root = Path(tissue_root)
        self.depth_root = Path(depth_root) if depth_root is not None else None
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.require_depth = bool(require_depth)
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        for relative_path, label in self.samples:
            rel = Path(relative_path)
            if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 2:
                raise ValueError(f"Expected stage_X/image path, got {relative_path!r}")
            if not 0 <= int(label) < 4:
                raise ValueError(f"Invalid stage index {label}: {relative_path}")
            for path in (self.data_root / rel, self.tissue_root / rel.with_suffix(".npy")):
                if not path.is_file():
                    raise FileNotFoundError(f"Required aligned input missing: {path}")
            if self.require_depth:
                if self.depth_root is None:
                    raise ValueError("require_depth=True needs a depth_root")
                path = self.depth_root / rel.with_suffix(".npy")
                if not path.is_file():
                    raise FileNotFoundError(f"Required relative-depth map missing: {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        relative_path, label = self.samples[index]
        rel = Path(relative_path)
        with Image.open(self.data_root / rel) as image:
            rgb_native = np.asarray(image.convert("RGB"), dtype=np.uint8)
        tissue_path = self.tissue_root / rel.with_suffix(".npy")
        cache = np.load(tissue_path, allow_pickle=False)
        if cache.dtype != np.uint8 or cache.ndim != 3 or cache.shape[2] != 8:
            raise ValueError(f"Expected HWC8 uint8 RGB+tissue cache: {tissue_path}")
        if cache.shape[:2] != rgb_native.shape[:2]:
            raise ValueError(f"Original/cached RGB shape mismatch: {tissue_path}")
        if not np.array_equal(cache[:, :, :3], rgb_native):
            raise ValueError(f"Original/cached RGB content mismatch: {tissue_path}")

        depth_path = self.depth_root / rel.with_suffix(".npy") if self.depth_root else None
        depth_valid = depth_path is not None and depth_path.is_file()
        if depth_valid:
            depth = np.load(depth_path, allow_pickle=False)
            validate_depth(depth, rgb_native.shape[:2], depth_path)
            depth = _resize_float(depth, self.image_size)
        else:
            if self.require_depth:
                raise FileNotFoundError(f"Required relative-depth map missing: {depth_path}")
            depth = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        rgb = np.asarray(Image.fromarray(rgb_native).resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        ), dtype=np.float32) / 255.0
        tissue = np.stack([
            _resize_float(cache[:, :, 3 + channel].astype(np.float32) / 255.0,
                          self.image_size)
            for channel in range(5)
        ], axis=-1)
        paired = np.concatenate((rgb, tissue, depth[:, :, None]), axis=-1)
        if self.augment:
            paired = _joint_augment(paired)
        rgb = (paired[:, :, :3] - IMAGENET_MEAN) / IMAGENET_STD

        def tensor(array):
            return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()

        return {
            "rgb": tensor(rgb),
            "tissue": tensor(paired[:, :, 3:8]),
            "depth": tensor(paired[:, :, 8:9]),
            "depth_valid": torch.tensor(depth_valid, dtype=torch.bool),
            "target": int(label),
            "path": str(relative_path),
        }
