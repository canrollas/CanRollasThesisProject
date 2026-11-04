"""Preserve the original four-stage folds, then filter and relabel each split."""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "task_3"))
from src.dataset import CLASS_ORDER, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.splits import build_folds  # noqa: E402

CLASSES = ["stage_2", "stage_3", "stage_4"]
CHANNEL_ORDER = ["R", "G", "B", "granulation", "slough", "necrosis", "tendon", "bone"]


def inspect_dataset(data_root):
    """Check every array header without loading full clinical photos into RAM.

    Fingerprints identify file membership, sizes, array headers and provenance;
    they are not a cryptographic checksum of every image pixel.
    """
    root = Path(data_root)
    records, counts = [], {}
    for stage in CLASS_ORDER:
        folder = root / stage
        if not folder.is_dir():
            raise ValueError(f"Missing {folder}. All four stage folders are needed to reproduce original folds.")
        files = sorted(p for p in folder.iterdir() if p.is_file())
        if not files:
            raise ValueError(f"Empty stage folder: {folder}")
        counts[stage] = len(files)
        for path in files:
            if path.suffix.lower() != ".npy":
                raise ValueError(f"Unexpected file in stage folder (only .npy allowed): {path}")
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 8 or min(arr.shape[:2]) < 1:
                raise ValueError(f"Expected uint8 H×W×8 RGB+tissue array: {path}, got {arr.shape}/{arr.dtype}")
            records.append([f"{stage}/{path.name}", path.stat().st_size, list(arr.shape), str(arr.dtype)])
            del arr
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if provenance is not None and provenance.get("channel_order", CHANNEL_ORDER) != CHANNEL_ORDER:
        raise ValueError("Dataset tissue channel order does not match Exp 3: " + str(provenance.get("channel_order")))
    payload = {"files": records, "source_provenance": provenance}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"fingerprint": digest, "class_counts": counts, "source_provenance": provenance}


def dataset_fingerprint(data_root):
    return inspect_dataset(data_root)["fingerprint"]


def build_stage234_folds(data_root, num_folds=3, val_fraction=0.1, seed=42):
    # Do NOT rebuild folds on a three-class directory or after relabeling:
    # Task 3 shuffles the full train/val pool, so that changes val membership.
    original = build_folds(data_root, num_folds, val_fraction, seed)
    folds = [
        {split: [(path, label - 1) for path, label in fold[split] if label != 0]
         for split in ("train", "val", "test")}
        for fold in original
    ]
    for fold_idx, fold in enumerate(folds):
        sets = {split: {path for path, _ in samples} for split, samples in fold.items()}
        if any(not samples for samples in fold.values()):
            raise ValueError(f"Fold {fold_idx} has an empty split after Stage 1 exclusion")
        if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
            raise ValueError(f"Overlapping samples in fold {fold_idx}")
    return folds


def make_manifest(folds):
    return {
        "original_classes": list(CLASS_ORDER), "classes": CLASSES,
        "folds": [
            {"fold": idx, **{
                split: [{"path": path, "label": label, "stage": CLASSES[label]} for path, label in samples]
                for split, samples in fold.items()
            }} for idx, fold in enumerate(folds)
        ],
    }


def balanced_subset(samples, per_class):
    return [sample for label in range(3) for sample in [s for s in samples if s[1] == label][:per_class]]


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class PairedStageDataset(Dataset):
    """Both models receive identical RGB; all spatial transforms are joint."""

    def __init__(self, samples, data_root, image_size, transform=None):
        self.samples = samples
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.transform = transform
        if any(label not in range(3) for _, label in samples):
            raise ValueError("Stage234 labels must be remapped to 0, 1, 2")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        arr = np.load(self.data_root / rel_path, allow_pickle=False)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 8:
            raise ValueError(f"Invalid eight-channel input: {rel_path}")
        size = self.image_size
        if arr.shape[:2] != (size, size):
            rgb = np.asarray(Image.fromarray(arr[:, :, :3]).resize((size, size), Image.Resampling.BILINEAR))
            tissue = np.stack([
                np.asarray(Image.fromarray(arr[:, :, c]).resize((size, size), Image.Resampling.BILINEAR))
                for c in range(3, 8)
            ], axis=-1)
            arr = np.concatenate([rgb, tissue], axis=-1)
        if self.transform is not None:
            arr = self.transform(arr)
        rgb = (arr[:, :, :3].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tissue = arr[:, :, 3:].astype(np.float32) / 255.0
        return (torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(tissue.transpose(2, 0, 1))), label)
