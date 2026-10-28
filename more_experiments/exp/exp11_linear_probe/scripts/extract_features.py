#!/usr/bin/env python3
"""Experiment 11 (Chapter 8): linear-probe diagnostic for CORN's errors
-- Step 2, feature extraction.

Question: does task_3's own trained MiT-B1+CORN encoder already carry
information that could separate the wounds CORN itself gets wrong, read
out with a different (linear) decision rule -- WITHOUT retraining or
touching the encoder at all? This script only runs frozen forward passes;
it never backpropagates into the encoder, and never loads gradients.

For one fold, loads task_3's own trained mit_b1__corn checkpoint
(task_3/results/task_3_outputs/checkpoints/mit_b1__corn_fold{fold}.pt),
and for every train/val/test image saves:

  final_features   [N, 512]  GAP(encoder's LAST feature map) -- the EXACT
                    vector CornHead itself reads. Backbone.forward does
                    this same pooling internally (src/model.py); this
                    script reaches into Backbone.encoder directly only to
                    ALSO keep the earlier stage3 map in the same pass.
  stage3_features   [N, 320]  GAP(encoder's THIRD-stage feature map,
                    [B,320,14,14]) -- an EARLIER representation, probed
                    as a control. task_3's CornHead is a single
                    `Linear(in_dim, num_classes-1)` with NO hidden layer
                    (verified directly in task_3/src/model.py) -- there is
                    no separate "pre-logit" layer distinct from
                    final_features to probe here, so stage3 substitutes
                    for it to ask the same "did discriminative
                    information get lost somewhere in the network"
                    question at an earlier point instead.
  labels            [N]  true stage index (0=stage_1 .. 3=stage_4)
  corn_preds        [N]  CORN's OWN prediction on this image, from this
                    EXACT checkpoint and this exact forward pass
                    (`predict_labels(logits, "corn")` on the real
                    CornHead's logits) -- not re-derived from any stored
                    summary, so it cannot silently drift from what this
                    script's final_features actually represent.
  ids               [N]  relative image path (e.g. "stage_3/foo.jpg"),
                    so probe predictions can be joined back to specific
                    images later (compare_errors.py, manual inspection).

A sanity check against task_3's own recorded ablation result is printed
(and must match, to float precision) -- this catches a silent key-split
or split-mismatch bug immediately, the same role exp7/exp10's "verify
reproduces baseline" checks play.

Written to results/fold_{fold}/{split}_{final_features,stage3_features,labels,corn_preds,ids}.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent  # exp/exp11_linear_probe/scripts/
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.dataset import StageDataset  # noqa: E402
from src.metrics import predict_labels  # noqa: E402
from src.model import Backbone, CornHead  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402


def load_checkpoint_into(backbone, head, checkpoint_path, device):
    """Same "0."/"1." key-split convention as exp5/6/7/10's model.py."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    backbone_state = {k[len("0."):]: v for k, v in state_dict.items() if k.startswith("0.")}
    head_state = {k[len("1."):]: v for k, v in state_dict.items() if k.startswith("1.")}
    if not backbone_state or not head_state:
        raise RuntimeError(
            f"{checkpoint_path}: expected keys prefixed '0.' (backbone) and '1.' (head), "
            f"got prefixes: {sorted({k.split('.')[0] for k in state_dict})}"
        )
    backbone.load_state_dict(backbone_state, strict=True)
    head.load_state_dict(head_state, strict=True)


@torch.no_grad()
def extract_split(backbone, head, samples, data_root, image_size, device, batch_size):
    ds = StageDataset(samples, data_root, image_size, transform=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    final_feats, stage3_feats, labels, corn_preds = [], [], [], []
    for images, targets in loader:
        images = images.to(device)
        feats = backbone.encoder(images)  # list of 6 stage feature maps (verified in exp7/exp10's model.py)
        stage3 = feats[-2].mean(dim=(2, 3))  # [B,320,14,14] -> [B,320]
        final = feats[-1].mean(dim=(2, 3))   # [B,512,7,7]   -> [B,512] -- identical op to Backbone.forward
        logits = head(final)
        preds = predict_labels(logits, "corn")

        final_feats.append(final.cpu().numpy())
        stage3_feats.append(stage3.cpu().numpy())
        labels.append(targets.numpy())
        corn_preds.append(preds.cpu().numpy())

    ids = np.array([s[0] for s in samples])
    return (
        np.concatenate(final_feats), np.concatenate(stage3_feats),
        np.concatenate(labels), np.concatenate(corn_preds), ids,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--batch", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = Path(args.data_root) if args.data_root else (REPO_ROOT / base_cfg["data_root"])
    batch_size = args.batch if args.batch is not None else base_cfg["batch_size"]
    num_classes = len(base_cfg["classes"])

    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    fold = folds[args.fold]

    checkpoint_path = REPO_ROOT / base_cfg["rgb_checkpoint_template"].format(fold=args.fold)
    if not checkpoint_path.is_file():
        raise SystemExit(f"BLOCKED: checkpoint not found: {checkpoint_path}")
    print(f"[fold{args.fold}] loading {checkpoint_path}", flush=True)

    device = get_device()
    backbone = Backbone(base_cfg["encoder"], None)
    head = CornHead(backbone.out_dim, num_classes)
    load_checkpoint_into(backbone, head, checkpoint_path, device)
    backbone.to(device).eval()
    head.to(device).eval()

    out_dir = EXP_DIR / base_cfg["results_dir"] / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        final, stage3, labels, corn_preds, ids = extract_split(
            backbone, head, fold[split], data_root, base_cfg["image_size"], device, batch_size,
        )
        np.save(out_dir / f"{split}_final_features.npy", final)
        np.save(out_dir / f"{split}_stage3_features.npy", stage3)
        np.save(out_dir / f"{split}_labels.npy", labels)
        np.save(out_dir / f"{split}_corn_preds.npy", corn_preds)
        np.save(out_dir / f"{split}_ids.npy", ids)
        acc = float((labels == corn_preds).mean())
        print(f"[fold{args.fold} {split}] n={len(labels)} corn_acc={acc:.4f} "
              f"final_dim={final.shape[1]} stage3_dim={stage3.shape[1]}", flush=True)

    print(f"[fold{args.fold}] wrote features to {out_dir}", flush=True)
    print(f"[fold{args.fold}] sanity check: compare the 'test' corn_acc line above against "
          f"task_3/results/task_3_outputs/results/ablation_results.csv's mit_b1__corn fold{args.fold} "
          f"row -- must match to 4 decimal places (same checkpoint, same split, same decode rule).",
          flush=True)


if __name__ == "__main__":
    main()
