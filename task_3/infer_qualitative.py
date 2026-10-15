#!/usr/bin/env python3
"""Ad hoc script (not part of the original ablation) to produce genuine
qualitative examples and a real confusion matrix for the thesis chapter,
using whichever checkpoints survived on disk. Verifies its own numbers
against results/task_3_outputs/results/ablation_results.csv before trusting
anything it prints.
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR))

from src.dataset import StageDataset, CLASS_ORDER
from src.engine import run_epoch
from src.losses import build_loss
from src.model import build_model
from src.splits import build_folds
from src.utils import get_device, load_yaml

ENCODER = sys.argv[1] if len(sys.argv) > 1 else "convnextv2_tiny"
FOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CKPT_DIR = TASK_DIR / "results" / "task_3_outputs" / "checkpoints"

def main():
    base_cfg = load_yaml(TASK_DIR / "configs" / "base.yaml")
    data_root = TASK_DIR.parent / "datasets" / "wound-stage-classification"
    device = get_device()
    num_classes = len(base_cfg["classes"])

    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    fold = folds[FOLD]
    test_samples = fold["test"]
    print(f"fold {FOLD} test set: {len(test_samples)} images")

    test_ds = StageDataset(test_samples, data_root, base_cfg["image_size"], transform=None)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    results = {}
    for head in ["softmax", "corn"]:
        ckpt_path = CKPT_DIR / f"{ENCODER}__{head}_fold{FOLD}.pt"
        model = build_model(ENCODER, None, head, num_classes=num_classes)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        criterion = build_loss(head, num_classes)
        stats = run_epoch(model, test_loader, criterion, device, head, num_classes, optimizer=None)
        print(f"\n=== {ENCODER} / {head} / fold{FOLD} (recomputed) ===")
        print(f"accuracy={stats['accuracy']:.4f} macro_f1={stats['macro_f1']:.4f} "
              f"mae={stats['mae']:.4f} qwk={stats['qwk']:.4f}")
        print("confusion matrix (rows=true, cols=pred), classes", CLASS_ORDER)
        print(stats["confusion"])
        results[head] = {"model": model, "stats": stats}

    # --- Per-image predictions for both heads, to pick qualitative examples ---
    from src.metrics import predict_labels
    import numpy as np
    from PIL import Image

    per_image = []
    for head in ["softmax", "corn"]:
        model = results[head]["model"]
        model.eval()
        preds_all = []
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                logits = model(images)
                preds = predict_labels(logits, head)
                preds_all.extend(preds.cpu().tolist())
        results[head]["preds"] = preds_all

    true_labels = [s[1] for s in test_samples]
    rel_paths = [s[0] for s in test_samples]
    sm_preds = results["softmax"]["preds"]
    corn_preds = results["corn"]["preds"]

    out_dir = TASK_DIR / "results" / "qual_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, (rel, y) in enumerate(zip(rel_paths, true_labels)):
        manifest.append({
            "rel_path": rel, "true": y, "true_stage": CLASS_ORDER[y],
            "softmax_pred": sm_preds[i], "softmax_pred_stage": CLASS_ORDER[sm_preds[i]],
            "corn_pred": corn_preds[i], "corn_pred_stage": CLASS_ORDER[corn_preds[i]],
        })

    manifest_path = out_dir / f"{ENCODER}_fold{FOLD}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote manifest with {len(manifest)} entries to {manifest_path}")

    # cross-check against the ablation CSV for this encoder/head/fold
    import csv
    csv_path = TASK_DIR / "results" / "task_3_outputs" / "results" / "ablation_results.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for head in ["softmax", "corn"]:
        row = [r for r in rows if r["encoder"] == ENCODER and r["head"] == head and r["fold"] == str(FOLD)][0]
        recomputed = results[head]["stats"]
        print(f"\ncheck {head}: csv acc={row['accuracy']} vs recomputed={recomputed['accuracy']:.4f} | "
              f"csv qwk={row['qwk']} vs recomputed={recomputed['qwk']:.4f}")

if __name__ == "__main__":
    main()
