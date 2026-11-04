"""Derive every aggregate from atomic, complete fold reports; never mix scopes."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from src.metrics import compute_metrics

METRICS = ("accuracy", "macro_f1", "mae", "qwk")
CLASSES = ("stage_2", "stage_3", "stage_4")


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_reports(output_dir):
    root = Path(output_dir) / "results"
    bundles = [json.loads(path.read_text()) for path in sorted((root / "folds").glob("*.json"))]
    if not bundles:
        return
    results, predictions = [], []
    for bundle in bundles:
        results.append({key: bundle[key] for key in (
            "config_id", "variant", "encoder", "head", "weights", "fold", "n_test", "params_m",
            *METRICS, "best_epoch", "best_val_macro_f1", "train_time_s",
        )})
        predictions.extend(bundle["predictions"])
    write_csv(root / "results.csv", results)
    write_csv(root / "test_predictions.csv", predictions)

    summaries, per_class, confusions, pooled = [], [], [], []
    for variant in sorted({row["variant"] for row in results}):
        rows = [row for row in results if row["variant"] == variant]
        summary = {"variant": variant, "n_folds": len(rows), "folds": ",".join(str(r["fold"]) for r in rows)}
        for key in METRICS:
            values = [r[key] for r in rows]
            summary.update({f"{key}_mean": float(np.mean(values)), f"{key}_std": float(np.std(values))})
        summaries.append(summary)
        scopes = [(str(row["fold"]), [p for p in predictions if p["variant"] == variant and p["fold"] == row["fold"]])
                  for row in rows]
        scopes.append(("pooled", [p for p in predictions if p["variant"] == variant]))
        for scope, items in scopes:
            true, pred = [p["true_index"] for p in items], [p["pred_index"] for p in items]
            stats = compute_metrics(true, pred, num_classes=3)
            if scope == "pooled":
                pooled.append({"variant": variant, "n_folds": len(rows), "n_test": len(items),
                               **{key: stats[key] for key in METRICS}})
            precision, recall, f1, support = precision_recall_fscore_support(true, pred, labels=[0, 1, 2], zero_division=0)
            for idx, stage in enumerate(CLASSES):
                per_class.append({"variant": variant, "fold": scope, "stage": stage,
                                  "precision": precision[idx], "recall": recall[idx], "f1": f1[idx], "support": support[idx]})
                for j, predicted_stage in enumerate(CLASSES):
                    confusions.append({"variant": variant, "fold": scope, "true_stage": stage,
                                       "pred_stage": predicted_stage, "count": int(stats["confusion"][idx, j])})
    write_csv(root / "summary.csv", summaries)
    write_csv(root / "pooled_metrics.csv", pooled)
    write_csv(root / "per_class_metrics.csv", per_class)
    write_csv(root / "confusion_matrices.csv", confusions)

    indexed = {(row["variant"], row["fold"]): row for row in results}
    matched_folds = sorted({fold for variant, fold in indexed if ("rgb", fold) in indexed and ("rgb_tissue", fold) in indexed})
    paired = []
    for fold in matched_folds:
        rgb, fusion = indexed["rgb", fold], indexed["rgb_tissue", fold]
        row = {"fold": str(fold), "n_paired_folds": 1}
        for key in METRICS:
            row.update({f"rgb_{key}": rgb[key], f"rgb_tissue_{key}": fusion[key],
                        f"delta_{key}": fusion[key] - rgb[key]})
        paired.append(row)
    if paired:
        paired.append({"fold": "mean", "n_paired_folds": len(matched_folds),
                       **{key: float(np.mean([r[key] for r in paired])) for key in paired[0] if key not in ("fold", "n_paired_folds")}})
        write_csv(root / "paired_comparison.csv", paired)
