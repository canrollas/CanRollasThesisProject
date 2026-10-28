#!/usr/bin/env python3
"""Compare separately trained RGB, tissue, geometry and full controls.

Reads saved evaluation outputs only. No model loading, training, inference,
or automatic test evaluation occurs. A single-seed target check is exploratory.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from evaluate import bootstrap_fold_mean, metrics, paired

MODES = ("rgb", "tissue", "geometry", "full")
METRICS = ("accuracy", "macro_f1", "qwk", "mae")
HISTORICAL_ACCURACY = 0.8168


def _read_run(root, mode, split):
    directory = Path(root) / mode / f"evaluation_{split}"
    for filename in ("metrics.json", "predictions.csv"):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"Required separately trained {mode} evaluation missing: {directory / filename}")
    report = json.loads((directory / "metrics.json").read_text())
    required = ("mode", "train_seed", "config", "code_fingerprint", "input_fingerprint",
                "split", "complete_three_fold", "smoke", "decoder", "folds")
    if any(key not in report for key in required):
        raise ValueError(f"Missing evaluation provenance for {mode}; rerun evaluate.py")
    if report["mode"] != mode or report["split"] != split:
        raise ValueError(f"Evaluation mode/split mismatch: {directory}")
    if report["smoke"] is not False:
        raise ValueError("Smoke evaluations cannot produce a production comparison")
    if not report["code_fingerprint"]:
        raise ValueError(f"Missing code fingerprint: {mode}")
    folds = {int(fold) for fold in report["folds"]}
    if not folds or not folds <= {0, 1, 2}:
        raise ValueError(f"Invalid evaluated fold membership: {mode}")
    if report["complete_three_fold"] is not (folds == {0, 1, 2}):
        raise ValueError(f"Incorrect complete_three_fold status: {mode}")
    rows = {condition: {fold: {} for fold in folds} for condition in ("real", "original_rgb")}
    with open(directory / "predictions.csv", newline="") as stream:
        for raw in csv.DictReader(stream):
            condition = raw["condition"]
            if condition not in rows:
                continue
            fold, target, prediction = int(raw["fold"]), int(raw["target"]), int(raw["prediction"])
            path = raw["path"]
            if fold not in folds or target not in range(4) or prediction not in range(4) or not path:
                raise ValueError(f"Invalid prediction record: {mode}")
            if path in rows[condition][fold]:
                raise ValueError(f"Duplicate prediction within fold: {mode}/{fold}/{condition}/{path}")
            rows[condition][fold][path] = {"path": path, "target": target, "prediction": prediction}
    for condition in rows:
        for fold in folds:
            if not rows[condition][fold]:
                raise ValueError(f"Missing predictions: {mode}/{fold}/{condition}")
            ordered = [rows[condition][fold][path] for path in sorted(rows[condition][fold])]
            actual = metrics(ordered)
            reported = report["folds"][str(fold)][condition]
            for key in METRICS:
                if not math.isclose(actual[key], reported[key], abs_tol=1e-9, rel_tol=1e-9):
                    raise ValueError(f"Saved metrics/predictions mismatch: {mode}/{fold}/{condition}/{key}")
            rows[condition][fold] = ordered
            if not report["folds"][str(fold)].get("baseline_sha256"):
                raise ValueError(f"Missing original RGB checkpoint fingerprint: {mode}/{fold}")
    return report, rows


def _membership(rows):
    return [(row["path"], row["target"]) for row in rows]


def _rollup(by_fold):
    fold_metrics = {str(fold): metrics(rows) for fold, rows in sorted(by_fold.items())}
    aggregate = {key: {"mean": float(np.mean([result[key] for result in fold_metrics.values()])),
                       "std": float(np.std([result[key] for result in fold_metrics.values()]))}
                 for key in METRICS}
    return {"folds": fold_metrics, "aggregate": aggregate}


def _comparison(reference, candidate, split):
    fold_reports = {}
    pairs = []
    for fold in sorted(reference):
        pair = (reference[fold], candidate[fold])
        result = paired(*pair)
        if split != "test":
            result["mcnemar_exact_two_sided_p"] = None
        fold_reports[str(fold)] = result
        pairs.append(pair)
    pooled = paired([row for before, _ in pairs for row in before],
                    [row for _, after in pairs for row in after])
    if split != "test":
        pooled["mcnemar_exact_two_sided_p"] = None
    return {"folds": fold_reports, "pooled": pooled,
            "accuracy_delta_fold_mean_image_bootstrap_95ci": bootstrap_fold_mean(pairs) if split == "test" else None,
            "inference_note": ("Paired image-level inference; patient independence and training-seed variability are not established"
                               if split == "test" else
                               "Disabled for validation: images can repeat across outer folds and models were selected on validation")}


def compare_runs(root, split="val"):
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    loaded = {mode: _read_run(root, mode, split) for mode in MODES}
    reference_report, reference_rows = loaded["full"]
    folds = set(reference_rows["real"])
    full_inputs = reference_report["input_fingerprint"]
    for mode, (report, rows) in loaded.items():
        for key in ("train_seed", "config", "code_fingerprint", "decoder", "complete_three_fold"):
            if report[key] != reference_report[key]:
                raise ValueError(f"Training configuration/provenance differs: {mode}/{key}")
        expected_inputs = {key: value for key, value in full_inputs.items()
                           if mode in ("geometry", "full") or not key.startswith("depth")}
        if not {"rgb", "tissue"} <= set(report["input_fingerprint"]):
            raise ValueError(f"Missing source input fingerprints: {mode}")
        if mode in ("geometry", "full") and "depth" not in report["input_fingerprint"]:
            raise ValueError(f"Missing depth input fingerprint: {mode}")
        if report["input_fingerprint"] != expected_inputs:
            raise ValueError(f"Modality cache fingerprints differ: {mode}")
        if set(rows["real"]) != folds:
            raise ValueError(f"Evaluated fold membership differs: {mode}")
        seen = {}
        for fold in sorted(folds):
            full_real = reference_rows["real"][fold]
            for condition in ("real", "original_rgb"):
                if _membership(rows[condition][fold]) != _membership(full_real):
                    raise ValueError(f"Paired image membership/targets differ: {mode}/{fold}/{condition}")
            if rows["original_rgb"][fold] != reference_rows["original_rgb"][fold]:
                raise ValueError(f"Original RGB predictions differ: {mode}/{fold}")
            if report["folds"][str(fold)]["baseline_sha256"] != reference_report["folds"][str(fold)]["baseline_sha256"]:
                raise ValueError(f"Original RGB checkpoint fingerprint differs: {mode}/{fold}")
            for row in rows["real"][fold]:
                if row["path"] in seen:
                    if split == "test":
                        raise ValueError(f"Test image occurs in multiple outer folds: {row['path']}")
                    if seen[row["path"]] != row["target"]:
                        raise ValueError(f"Repeated validation image has inconsistent targets: {row['path']}")
                seen[row["path"]] = row["target"]
    rollups = {mode: _rollup(rows["real"]) for mode, (_, rows) in loaded.items()}
    rollups["original_rgb"] = _rollup(reference_rows["original_rgb"])
    comparisons = {mode: _comparison(rows["real"], reference_rows["real"], split)
                   for mode, (_, rows) in loaded.items() if mode != "full"}
    comparisons["original_rgb"] = _comparison(reference_rows["original_rgb"], reference_rows["real"], split)
    eligible = split == "test" and folds == {0, 1, 2}
    candidate_accuracy = rollups["full"]["aggregate"]["accuracy"]["mean"]
    thresholds = {"historical_rgb": HISTORICAL_ACCURACY,
                  "remeasured_original_rgb": rollups["original_rgb"]["aggregate"]["accuracy"]["mean"],
                  "separately_trained_rgb": rollups["rgb"]["aggregate"]["accuracy"]["mean"]}
    strictly_higher = {name: candidate_accuracy > value for name, value in thresholds.items()}
    target_check = {"eligible": eligible,
                    "status": ("met_for_this_seed" if all(strictly_higher.values()) else "not_met") if eligible else "not_eligible",
                    "full_accuracy_fold_mean": candidate_accuracy, "thresholds": thresholds,
                    "strictly_higher": strictly_higher if eligible else None,
                    "interpretation": "Exploratory single-training-seed accuracy comparison; no statistical significance or clinical validity claim"}
    paths = [row["path"] for values in reference_rows["real"].values() for row in values]
    return {"split": split, "train_seed": reference_report["train_seed"], "training_seeds_evaluated": 1,
            "complete_three_fold": folds == {0, 1, 2}, "smoke": False,
            "historical_rgb_accuracy_mean": HISTORICAL_ACCURACY,
            "prediction_records": len(paths), "unique_images": len(set(paths)),
            "repeated_validation_records": len(paths) - len(set(paths)) if split == "val" else 0,
            "modes": rollups, "full_vs_separately_trained_controls": comparisons,
            "target_check": target_check,
            "caveat": "Single training seed; image-level evaluation. Patient independence, clinical validity, and robustness across seeds are not established."}


def write_comparison(result, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = ["mode", "split", "train_seed", "n_folds"]
    fields += [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    with open(output / "summary.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode, rollup in result["modes"].items():
            writer.writerow({"mode": mode, "split": result["split"], "train_seed": result["train_seed"],
                             "n_folds": len(rollup["folds"]),
                             **{f"{metric}_{stat}": rollup["aggregate"][metric][stat]
                                for metric in METRICS for stat in ("mean", "std")}})
        writer.writerow({"mode": "historical_rgb", "split": "historical_test", "n_folds": 3,
                         "accuracy_mean": HISTORICAL_ACCURACY})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Parent of rgb/, tissue/, geometry/, full/, e.g. runs/main/seed42")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True, help="New unused output directory")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {args.output_dir}")
    result = compare_runs(args.root, args.split)
    write_comparison(result, args.output_dir)
    print(json.dumps({"target_check": result["target_check"], "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
