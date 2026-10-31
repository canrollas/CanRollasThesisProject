#!/usr/bin/env python3
"""Work Package 4, step 2: review panels for manual inspection.

For every error_3to4 / error_4to3 case (all of them -- 102 total) and a
seeded random sample of correct_3 / correct_4 cases (comparison set, default
20 each), renders one PNG: the original RGB photo, the bone-probability
channel overlay, and the tendon-probability channel overlay, side by side,
titled with true/predicted stage, fold, and the RGB baseline's own decoded
probability (from stage34_error_cases.csv, which already carries it -- no
model is loaded here either).

Pure visualization of already-computed data -- draws no conclusion, assigns
no label.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
WP_DIR = THIS_DIR.parent
EXT_DIR = WP_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]

CASES_CSV = EXT_DIR / "stage34_error_cases.csv"
RGB_DATA_ROOT = REPO_ROOT / "datasets/wound-stage-classification"
TISSUE_DATA_ROOT = REPO_ROOT / "datasets/wound-staging-masks-exp-3"
OUT_DIR = EXT_DIR / "error_panels"


def render_one(row, out_path):
    rel_path = Path(row["path"])
    rgb = np.array(Image.open(RGB_DATA_ROOT / rel_path).convert("RGB"))
    arr = np.load(TISSUE_DATA_ROOT / rel_path.with_suffix(".npy"))
    tendon = arr[:, :, 6].astype(np.float32) / 255.0
    bone = arr[:, :, 7].astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(rgb)
    axes[1].imshow(bone, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title(f"bone (mean={row['bone_mean']}, max={row['bone_max']})")
    axes[2].imshow(rgb)
    axes[2].imshow(tendon, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title(f"tendon (mean={row['tendon_mean']}, max={row['tendon_max']})")
    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{row['path']}  fold={row['fold']}  true={row['true_stage']}  pred={row['pred_stage']}  "
        f"P(pred)={float(row['predicted_probability']):.3f}  entropy={float(row['entropy']):.3f}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-correct-per-group", type=int, default=20,
                    help="comparison-set sample size for correct_3/correct_4 (all errors are always rendered)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not CASES_CSV.is_file():
        raise SystemExit(f"BLOCKED: {CASES_CSV} not found -- run build_case_table.py first.")

    with open(CASES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)

    rng = random.Random(args.seed)
    to_render = list(by_group.get("error_3to4", [])) + list(by_group.get("error_4to3", []))
    for group in ("correct_3", "correct_4"):
        pool = by_group.get(group, [])
        sample = pool if len(pool) <= args.n_correct_per_group else rng.sample(pool, args.n_correct_per_group)
        to_render.extend(sample)

    print(f"rendering {len(to_render)} panels "
          f"(all errors: {len(by_group.get('error_3to4', [])) + len(by_group.get('error_4to3', []))}, "
          f"+ up to {args.n_correct_per_group} each of correct_3/correct_4)")

    n_ok, n_fail = 0, 0
    for row in to_render:
        stem = Path(row["path"]).stem
        out_path = OUT_DIR / row["group"] / f"{stem}.png"
        try:
            render_one(row, out_path)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - one broken image shouldn't stop the batch
            print(f"FAILED {row['path']}: {exc}", file=sys.stderr)
            n_fail += 1

    print(f"wrote {n_ok} panels to {OUT_DIR} ({n_fail} failed)")


if __name__ == "__main__":
    main()
