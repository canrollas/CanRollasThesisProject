#!/usr/bin/env python3
"""Experiment 9 Step 1 -- cache frozen-teacher global embeddings.

Later steps will train a single student to regress multiple teachers'
pooled embeddings (feature-level multi-teacher distillation), then attach
a CORN head on the distilled representation. This script only does the
cheap, non-trainable part: run each already-trained teacher backbone over
every image once (no augmentation, deterministic) and cache the pooled
embedding to disk, so the distillation step never has to load a teacher
or touch a raw image again.

Leakage note: there is no single "the mit_b1 teacher" -- each encoder has
3 independent checkpoints (mit_b1__corn_fold{0,1,2}.pt), each trained
excluding its own fold. A fold's student must only ever see embeddings
produced by that SAME fold's teacher checkpoint (train images included --
that teacher trained on them, which is fine -- and test images too, which
is exactly the already-established leak-free eval protocol every other
experiment in this repo uses). So this caches one embedding matrix PER
(teacher, fold) pair, not one per teacher; which file a given fold's
student reads from is a later-step concern, not resolved here.

TEACHERS below only includes encoders with all 3 CORN-head fold
checkpoints present AND state-dict-compatible with the current
src/model.py (verified by the strict=True load in build_teacher below).
convnextv2_tiny is excluded for now -- its only available fold2
checkpoint was trained with a different, no-longer-matching model.py
(extra norm layer between backbone and head) and fails a strict load.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
EXP9_DIR = THIS_DIR.parent
REPO_ROOT = EXP9_DIR.parents[2]
TASK3_DIR = REPO_ROOT / "task_3"

sys.path.insert(0, str(TASK3_DIR))
from src.dataset import CLASS_ORDER, StageDataset, list_samples  # noqa: E402
from src.model import Backbone  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device  # noqa: E402

NUM_FOLDS = 3
SEED = 42
VAL_FRACTION = 0.1
IMAGE_SIZE = 224
BATCH_SIZE = 16

CHECKPOINT_DIR = TASK3_DIR / "results" / "task_3_outputs" / "checkpoints"
TEACHERS = {
    "mit_b1": CHECKPOINT_DIR / "mit_b1__corn_fold{fold}.pt",
    "resnext50_32x4d": CHECKPOINT_DIR / "resnext50_32x4d__corn_fold{fold}.pt",
    # fold2 here only strict-loads on its Backbone half (see the .provenance.json
    # next to it) -- build_teacher() below only ever reads the "0."-prefixed
    # backbone keys anyway, so that's exactly what this needs.
    "convnextv2_tiny": CHECKPOINT_DIR / "convnextv2_tiny__corn_fold{fold}.pt",
}

FEATURES_DIR = EXP9_DIR / "features"


def build_index(data_root):
    """Canonical, fixed row order (sorted per class, same order
    list_samples()/StageDataset already use everywhere else in this repo)
    plus each image's permanent CV test-fold membership."""
    samples = list_samples(data_root)  # [(rel_path, label), ...], canonical order
    folds = build_folds(data_root, NUM_FOLDS, VAL_FRACTION, SEED)

    fold_of = {}
    for fold_idx, fold in enumerate(folds):
        for rel_path, _ in fold["test"]:
            fold_of[rel_path] = fold_idx

    index_rows = []
    for i, (rel_path, label) in enumerate(samples):
        index_rows.append({
            "idx": i,
            "stem": Path(rel_path).stem,
            "rel_path": rel_path,
            "label": label,
            "stage_name": CLASS_ORDER[label],
            "cv_fold": fold_of[rel_path],
        })
    return samples, index_rows


def write_index_csv(index_rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "stem", "rel_path", "label", "stage_name", "cv_fold"])
        writer.writeheader()
        writer.writerows(index_rows)


def build_teacher(encoder_name, checkpoint_path, device):
    backbone = Backbone(encoder_name, None)
    full_state = torch.load(checkpoint_path, map_location="cpu")
    backbone_state = {k[len("0."):]: v for k, v in full_state.items() if k.startswith("0.")}
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.to(device)
    backbone.eval()
    return backbone


@torch.no_grad()
def extract_embeddings(backbone, samples, data_root, device):
    ds = StageDataset(samples, data_root, IMAGE_SIZE, transform=None)
    # pin_memory is CUDA-only and MPS warns if asked for it -- skip it rather
    # than fall back to CPU, which was ~10x slower on this machine.
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                         pin_memory=(device.type == "cuda"))
    chunks = []
    for images, _ in loader:
        out = backbone(images.to(device))
        chunks.append(out.cpu().numpy().astype(np.float16))
    return np.concatenate(chunks, axis=0)


def main():
    device = get_device()
    print(f"device={device}", flush=True)
    data_root = REPO_ROOT / "datasets" / "wound-stage-classification"
    samples, index_rows = build_index(data_root)
    write_index_csv(index_rows, FEATURES_DIR / "index.csv")
    print(f"index: {len(samples)} images -> {FEATURES_DIR / 'index.csv'}")

    for teacher_name, ckpt_template in TEACHERS.items():
        for fold_idx in range(NUM_FOLDS):
            out_path = FEATURES_DIR / f"{teacher_name}_fold{fold_idx}.npy"
            if out_path.exists():
                print(f"{teacher_name} fold{fold_idx}: already cached, skipping ({out_path})")
                continue

            ckpt_path = Path(str(ckpt_template).format(fold=fold_idx))
            if not ckpt_path.exists():
                print(f"SKIP {teacher_name} fold{fold_idx}: checkpoint missing ({ckpt_path})")
                continue

            backbone = build_teacher(teacher_name, ckpt_path, device)
            embeddings = extract_embeddings(backbone, samples, data_root, device)
            np.save(out_path, embeddings)
            print(f"{teacher_name} fold{fold_idx}: {embeddings.shape} {embeddings.dtype} -> {out_path}")
            del backbone

    print("done")


if __name__ == "__main__":
    main()
