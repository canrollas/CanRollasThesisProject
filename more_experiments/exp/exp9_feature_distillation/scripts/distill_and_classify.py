#!/usr/bin/env python3
"""Experiment 9, Steps 2+3 -- multi-teacher feature distillation, then CORN.

Step 2 (distillation pretraining): a single student backbone
(efficientnet-b0 -- deliberately NOT one of the 3 teachers, so it isn't
just re-deriving one teacher's own features) learns, from the raw image
alone, to reproduce all 3 frozen teachers' pooled embeddings
(mit_b1 512-dim, resnext50_32x4d 2048-dim, convnextv2_tiny 768-dim) at
once. One small linear projection head per teacher maps the student's
320-dim feature into that teacher's embedding space; loss is the average
per-teacher cosine distance (robust to the 3 teachers' very different
embedding scales, unlike raw MSE). No classification labels are used in
this phase -- it only needs the cached teacher embeddings from Step 1.

Step 3 (CORN classification): the distilled backbone is frozen (Phase A --
linear probe, keeps the trainable-parameter count tiny given 1091 images)
and a fresh CornHead is trained on top with real stage labels, reusing
task_3/src/engine.py's fit()/run_epoch() unchanged (same optimizer,
scheduler, loss, and metrics every other config in this repo uses, so the
result slots directly into the existing comparison tables).

Same build_folds(seed=42) split as everywhere else; for fold i, every
teacher-embedding target comes from that teacher's fold-i checkpoint
(which excluded fold i during its own training), so neither phase ever
leaks fold i's test images.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
EXP9_DIR = THIS_DIR.parent
REPO_ROOT = EXP9_DIR.parents[2]
TASK3_DIR = REPO_ROOT / "task_3"

sys.path.insert(0, str(TASK3_DIR))
from src.augmentations import ImageAugmentation  # noqa: E402
from src.dataset import StageDataset  # noqa: E402
from src.engine import fit, run_epoch  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.model import Backbone, CornHead  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml, set_seed  # noqa: E402

NUM_FOLDS = 3
SEED = 42
VAL_FRACTION = 0.1
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 4

STUDENT_ENCODER = "efficientnet-b0"  # not one of the 3 teachers on purpose
TEACHER_DIMS = {"mit_b1": 512, "resnext50_32x4d": 2048, "convnextv2_tiny": 768}

FEATURES_DIR = EXP9_DIR / "features"
CHECKPOINT_DIR = EXP9_DIR / "checkpoints"
RESULTS_CSV = EXP9_DIR / "results" / "exp9_results.csv"

DISTILL_EPOCHS = 60
DISTILL_LR = 1e-4
DISTILL_WEIGHT_DECAY = 1e-5


def set_distill_epochs(n):
    global DISTILL_EPOCHS
    DISTILL_EPOCHS = n


# ---------------------------------------------------------------------------
# Step 2: distillation
# ---------------------------------------------------------------------------

class ProjectionHeads(nn.Module):
    def __init__(self, student_dim, teacher_dims):
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Linear(student_dim, dim) for name, dim in teacher_dims.items()})

    def forward(self, student_feat):
        return {name: head(student_feat) for name, head in self.heads.items()}


def cosine_distill_loss(projected, targets):
    losses = [1 - F.cosine_similarity(projected[name], targets[name], dim=1).mean() for name in projected]
    return torch.stack(losses).mean(), {name: l.item() for name, l in zip(projected, losses)}


class DistillDataset(Dataset):
    """(image, {teacher_name: target_embedding}) -- no labels, Step 2 doesn't need them."""

    def __init__(self, samples, data_root, stem_to_row, teacher_arrays, transform=None):
        self.samples = samples
        self.data_root = Path(data_root)
        self.stem_to_row = stem_to_row
        self.teacher_arrays = teacher_arrays
        self.transform = transform
        self.image_size = IMAGE_SIZE

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, _ = self.samples[idx]
        from PIL import Image
        from src.dataset import IMAGENET_MEAN, IMAGENET_STD
        image = Image.open(self.data_root / rel_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.array(image, dtype=np.uint8)
        if self.transform is not None:
            image_np = self.transform(image_np)
        image_f = image_np.astype(np.float32) / 255.0
        image_f = (image_f - IMAGENET_MEAN) / IMAGENET_STD
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).float()

        row = self.stem_to_row[Path(rel_path).stem]
        targets = {name: torch.from_numpy(arr[row].astype(np.float32)) for name, arr in self.teacher_arrays.items()}
        return image_t, targets


def collate_distill(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {name: torch.stack([b[1][name] for b in batch]) for name in batch[0][1]}
    return images, targets


def load_index_and_arrays(fold_idx):
    stem_to_row = {}
    with open(FEATURES_DIR / "index.csv", newline="") as f:
        for row in csv.DictReader(f):
            stem_to_row[row["stem"]] = int(row["idx"])
    arrays = {name: np.load(FEATURES_DIR / f"{name}_fold{fold_idx}.npy") for name in TEACHER_DIMS}
    return stem_to_row, arrays


def run_distillation(fold_idx, train_samples, val_samples, data_root, device, log_prefix):
    stem_to_row, arrays = load_index_and_arrays(fold_idx)
    aug = ImageAugmentation(hflip_p=0.5, vflip_p=0.5, rotate_p=0.7, rotate_angles=(90, 180, 270))

    train_ds = DistillDataset(train_samples, data_root, stem_to_row, arrays, transform=aug)
    val_ds = DistillDataset(val_samples, data_root, stem_to_row, arrays, transform=None)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_distill)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_distill)

    backbone = Backbone(STUDENT_ENCODER, "imagenet").to(device)
    proj = ProjectionHeads(backbone.out_dim, TEACHER_DIMS).to(device)
    params = list(backbone.parameters()) + list(proj.parameters())
    optimizer = AdamW(params, lr=DISTILL_LR, weight_decay=DISTILL_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=DISTILL_EPOCHS, eta_min=DISTILL_LR * 0.01)

    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, DISTILL_EPOCHS + 1):
        backbone.train()
        proj.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            feat = backbone(images)
            projected = proj(feat)
            loss, _ = cosine_distill_loss(projected, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        backbone.eval()
        proj.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                feat = backbone(images)
                projected = proj(feat)
                loss, _ = cosine_distill_loss(projected, targets)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"{log_prefix} distill epoch {epoch}/{DISTILL_EPOCHS} "
                  f"train_cos_loss={train_loss:.4f} val_cos_loss={val_loss:.4f} best={best_val_loss:.4f}", flush=True)

    backbone.load_state_dict(best_state)
    return backbone, best_val_loss


# ---------------------------------------------------------------------------
# Step 3: CORN classification on top of the frozen distilled backbone
# ---------------------------------------------------------------------------

def run_classification(fold_idx, distilled_backbone, fold, data_root, base_cfg, device, log_prefix):
    for p in distilled_backbone.parameters():
        p.requires_grad = False

    head = CornHead(distilled_backbone.out_dim, NUM_CLASSES)
    model = nn.Sequential(distilled_backbone, head)

    aug = ImageAugmentation(**base_cfg["augment"])
    train_ds = StageDataset(fold["train"], data_root, IMAGE_SIZE, transform=aug)
    val_ds = StageDataset(fold["val"], data_root, IMAGE_SIZE, transform=None)
    test_ds = StageDataset(fold["test"], data_root, IMAGE_SIZE, transform=None)
    train_loader = DataLoader(train_ds, batch_size=base_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=base_cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=base_cfg["batch_size"], shuffle=False)

    checkpoint_path = CHECKPOINT_DIR / f"distilled_student__corn_fold{fold_idx}.pt"
    model, best_val_f1 = fit(
        model, train_loader, val_loader, device,
        epochs=base_cfg["epochs"], lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"],
        eta_min_factor=base_cfg["eta_min_factor"], head="corn", num_classes=NUM_CLASSES,
        log_prefix=log_prefix, checkpoint_path=checkpoint_path,
    )

    criterion = build_loss("corn", NUM_CLASSES)
    test_stats = run_epoch(model, test_loader, criterion, device, "corn", NUM_CLASSES, optimizer=None)
    return test_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--smoke", action="store_true", help="2 distill epochs, 2 CORN epochs, tiny subset")
    args = parser.parse_args()
    fold_indices = [int(f) for f in args.folds.split(",")]

    base_cfg = load_yaml(TASK3_DIR / "configs" / "base.yaml")
    if args.smoke:
        set_distill_epochs(2)
        base_cfg = dict(base_cfg, epochs=2)
    data_root = REPO_ROOT / "datasets" / "wound-stage-classification"
    device = get_device()
    print(f"device={device} student={STUDENT_ENCODER} teachers={list(TEACHER_DIMS)}", flush=True)

    set_seed(SEED)
    folds = build_folds(data_root, NUM_FOLDS, VAL_FRACTION, SEED)
    if args.smoke:
        folds = [{k: v[:8] for k, v in fold.items()} for fold in folds]

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["fold", "distill_val_cos_loss", "accuracy", "macro_f1", "mae", "qwk"])

        for fold_idx in fold_indices:
            fold = folds[fold_idx]
            log_prefix = f"[exp9 fold{fold_idx}]"
            print(f"{log_prefix} Step 2: distillation pretraining "
                  f"({len(fold['train'])} train / {len(fold['val'])} val)", flush=True)
            backbone, best_val_loss = run_distillation(
                fold_idx, fold["train"], fold["val"], data_root, device, log_prefix)

            print(f"{log_prefix} Step 3: CORN classification on frozen distilled backbone", flush=True)
            test_stats = run_classification(fold_idx, backbone, fold, data_root, base_cfg, device, log_prefix)

            print(f"{log_prefix} DONE distill_val_loss={best_val_loss:.4f} "
                  f"test_acc={test_stats['accuracy']:.4f} test_f1={test_stats['macro_f1']:.4f} "
                  f"test_mae={test_stats['mae']:.4f} test_qwk={test_stats['qwk']:.4f}", flush=True)

            writer.writerow([fold_idx, round(best_val_loss, 4), round(test_stats["accuracy"], 4),
                              round(test_stats["macro_f1"], 4), round(test_stats["mae"], 4),
                              round(test_stats["qwk"], 4)])
            f.flush()


if __name__ == "__main__":
    main()
