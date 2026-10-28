#!/usr/bin/env python3
"""Post-training diagnostic for exp-10's "attention"/"contrastive" modes:
directly checks the risk the architecture proposal itself names --
"Üç ayrı attention modülü aynı özellikleri öğrenebilir" (the three
attention modules might all converge to the same thing, making the
per-boundary design pointless even if accuracy doesn't regress).

For every test image in a fold, computes the three boundaries' attention
maps A_0, A_1, A_2 (model.py's BoundaryResidualHead.attention_map) and
reports, pairwise:
  - cosine similarity between the flattened maps (1.0 = identical)
  - Pearson correlation between the flattened maps
  - "argmax agreement": whether the two maps' single highest-attention
    cell is the same 7x7 location

High similarity/correlation and high argmax agreement across all three
boundary pairs would mean the attention modules learned near-identical
spatial focus -- the architecture's premise (boundary-specific spatial
selection) would not actually hold, independent of whatever the test
accuracy says. This is a diagnostic, not a success criterion by itself;
report both this and the accuracy table together.
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.dataset import StageDataset  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from model import build_model  # noqa: E402


def pairwise_stats(maps):
    """maps: dict[boundary_idx] -> np.ndarray [N, H*W] (flattened, one row
    per test image). Returns per-pair cosine/correlation/argmax-agreement,
    averaged over images."""
    results = {}
    for i, j in combinations(maps, 2):
        a, b = maps[i], maps[j]
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        cosine = (a_norm * b_norm).sum(axis=1)

        a_centered = a - a.mean(axis=1, keepdims=True)
        b_centered = b - b.mean(axis=1, keepdims=True)
        denom = (np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1)) + 1e-8
        correlation = (a_centered * b_centered).sum(axis=1) / denom

        argmax_agreement = (a.argmax(axis=1) == b.argmax(axis=1)).astype(np.float64)

        results[f"boundary{i}_vs_boundary{j}"] = {
            "mean_cosine": float(cosine.mean()), "std_cosine": float(cosine.std()),
            "mean_correlation": float(correlation.mean()),
            "argmax_agreement_rate": float(argmax_agreement.mean()),
            "n_images": int(len(a)),
        }
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("attention", "contrastive"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_yaml(EXP_DIR / "configs" / "base.yaml")
    data_root = Path(args.data_root) if args.data_root else EXP_DIR / base_cfg["data_root"]
    num_classes = len(base_cfg["classes"])

    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    samples = folds[args.fold][args.split]

    checkpoint_path = EXP_DIR / base_cfg["checkpoint_dir"] / f"mit_b1__corn__boundary_{args.mode}_fold{args.fold}.pt"
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"BLOCKED: no trained checkpoint at {checkpoint_path} -- run scripts/train.py "
            f"--mode {args.mode} --folds {args.fold} first."
        )

    device = get_device()
    model = build_model(base_cfg["encoder"], None, num_classes=num_classes,
                         hidden_dim=base_cfg["hidden_dim"], mode=args.mode)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    dataset = StageDataset(samples, data_root, base_cfg["image_size"], transform=None)
    loader = DataLoader(dataset, batch_size=base_cfg["batch_size"], shuffle=False,
                         num_workers=args.workers)

    num_boundaries = num_classes - 1
    maps = {k: [] for k in range(num_boundaries)}
    with torch.no_grad():
        for images, _labels in loader:
            images = images.to(device)
            feature_map = model.backbone(images)
            for k, boundary_head in enumerate(model.head.boundary_heads):
                attn = boundary_head.attention_map(feature_map)  # [B,1,H,W]
                maps[k].append(attn.flatten(1).cpu().numpy())
    maps = {k: np.concatenate(v, axis=0) for k, v in maps.items()}

    stats = pairwise_stats(maps)
    print(f"[{args.mode} fold{args.fold} {args.split}] n_images={len(samples)}")
    for pair, values in stats.items():
        print(f"  {pair}: cosine={values['mean_cosine']:.4f}±{values['std_cosine']:.4f} "
              f"corr={values['mean_correlation']:.4f} argmax_agreement={values['argmax_agreement_rate']:.4f}")

    out_path = EXP_DIR / "results" / f"attention_diagnostics_{args.mode}_fold{args.fold}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"mode": args.mode, "fold": args.fold, "split": args.split, "n_images": len(samples),
                    "pairwise": stats}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
