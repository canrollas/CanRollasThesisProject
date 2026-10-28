#!/usr/bin/env python3
"""Paired evaluation and deterministic prior ablations after validation selection.

Default is validation. Use --split test ONCE after fixing the design. All
predictions use task_3's ordinal-median CORN decoder, including original RGB.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader, Dataset

from train import (EXP_DIR, build_folds, compute_metrics, forward_batch, get_device,
                   model_from_config, move_batch, predict_labels, sha256, split_digest,
                   input_fingerprint, code_fingerprint, validate_baseline_protocol)
from data import StagePriorDataset


class PriorAblation(Dataset):
    """A dataset-level cyclic derangement; no self-pairs or batch-size effects."""

    def __init__(self, base, condition, seed):
        self.base, self.condition = base, condition
        n = len(base)
        self.donors = np.arange(n)
        if n > 1:
            order = np.random.default_rng(seed).permutation(n)
            self.donors[order] = np.roll(order, 1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = dict(self.base[idx])
        if self.condition.startswith("shuffle"):
            if len(self) < 2:
                raise ValueError("Derangement needs at least two images")
            donor = self.base[int(self.donors[idx])]
            if self.condition in ("shuffle_tissue", "shuffle_both"):
                sample["tissue"] = donor["tissue"]
            if self.condition in ("shuffle_depth", "shuffle_both"):
                sample["depth"], sample["depth_valid"] = donor["depth"], donor["depth_valid"]
        elif self.condition == "zero_tissue":
            sample["tissue"] = torch.zeros_like(sample["tissue"])
        elif self.condition == "uniform_tissue":
            support = sample["tissue"].sum(0, keepdim=True).clamp(0, 1)
            sample["tissue"] = support.expand_as(sample["tissue"]) / 5
        elif self.condition == "flat_depth":
            sample["depth"] = torch.zeros_like(sample["depth"])
        return sample


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    rows = []
    for raw in loader:
        batch = move_batch(raw, device)
        out = forward_batch(model, batch)
        pred = predict_labels(out["logits"], "corn").cpu().tolist()
        global_pred = predict_labels(out["rgb_logits"], "corn").cpu().tolist()
        for i, path in enumerate(batch["path"]):
            rows.append({"path": path, "target": int(batch["target"][i]),
                         "prediction": pred[i], "global_prediction": global_pred[i],
                         "geometry_valid": bool(out["geometry_valid"][i]),
                         "has_wound": bool(out["has_wound"][i]),
                         "logits": out["logits"][i].cpu().tolist()})
    return rows


def metrics(rows, key="prediction"):
    result = compute_metrics([r["target"] for r in rows], [r[key] for r in rows], 4)
    result["confusion"] = result["confusion"].tolist()
    return result


def paired(reference, candidate):
    if [r["path"] for r in reference] != [r["path"] for r in candidate]:
        raise ValueError("Paired comparison order differs")
    original = np.array([r["prediction"] == r["target"] for r in reference])
    proposed = np.array([r["prediction"] == r["target"] for r in candidate])
    fixes, harms = int((~original & proposed).sum()), int((original & ~proposed).sum())
    flips = sum(a["prediction"] != b["prediction"] for a, b in zip(reference, candidate))
    return {"n": len(reference), "fixes": fixes, "harms": harms, "net_correct": fixes - harms,
            "prediction_flips": flips, "accuracy_delta": float(proposed.mean() - original.mean()),
            "mcnemar_exact_two_sided_p": float(binomtest(fixes, fixes + harms).pvalue) if fixes + harms else 1.0}


def bootstrap_fold_mean(pairs, seed=42, repetitions=2000):
    """Paired image resampling within fold; NOT a patient-level confidence interval."""
    rng = np.random.default_rng(seed)
    draws = np.zeros(repetitions)
    for reference, candidate in pairs:
        delta = np.array([int(b["prediction"] == b["target"]) - int(a["prediction"] == a["target"])
                          for a, b in zip(reference, candidate)])
        for i in range(repetitions):
            draws[i] += rng.choice(delta, size=len(delta), replace=True).mean() / len(pairs)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing fold0.pt etc.")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--tissue-root", type=Path)
    parser.add_argument("--depth-root", type=Path)
    parser.add_argument("--rgb-checkpoint-template")
    args = parser.parse_args()
    indices = [int(f) for f in args.folds.split(",")]
    if len(indices) != len(set(indices)) or any(f not in range(3) for f in indices):
        raise ValueError("folds must be unique indices from 0,1,2")
    out_dir = args.output_dir or args.run_dir / f"evaluation_{args.split}"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite previous evaluation: {out_dir}")
    device = torch.device(args.device) if args.device else get_device()
    all_rows, fold_reports, pairs = [], {}, []
    run_identity = None
    verified_inputs = None
    for idx in indices:
        path = args.run_dir / f"fold{idx}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        meta = saved["fingerprint"]
        if meta["code"] != code_fingerprint():
            raise ValueError("Training/model/data code differs from selected checkpoint")
        if meta["fold"] != idx:
            raise ValueError("Checkpoint fold mismatch")
        if meta["smoke"] and args.split == "test":
            raise ValueError("Smoke checkpoints cannot produce official test results")
        identity = {k: v for k, v in meta.items() if k not in ("fold", "split_sha256", "baseline_sha256")}
        if run_identity is not None and run_identity != identity:
            raise ValueError("Checkpoints do not belong to the same training configuration")
        run_identity = identity
        cfg, mode = dict(meta["config"]), meta["mode"]
        for name in ("data_root", "tissue_root", "depth_root"):
            value = getattr(args, name)
            if value is not None:
                cfg[name] = str(value.resolve())
        if args.rgb_checkpoint_template:
            cfg["rgb_checkpoint_template"] = args.rgb_checkpoint_template
        folds = build_folds(cfg["data_root"], cfg["num_folds"], cfg["val_fraction"], cfg["split_seed"])
        validate_baseline_protocol(cfg, folds)
        if verified_inputs is None:
            all_samples = [s for f in folds for s in f["test"]]
            verified_inputs = input_fingerprint(cfg, all_samples, mode)
        if verified_inputs != meta["inputs"]:
            raise ValueError("Source images, auxiliary cache bytes or provenance differ from training")
        fold = folds[idx]
        if split_digest(fold) != meta["split_sha256"]:
            raise ValueError("Evaluation split differs from training")
        samples = fold[args.split][:8] if meta["smoke"] else fold[args.split]
        base = StagePriorDataset(samples, cfg["data_root"], cfg["tissue_root"],
                                 cfg["depth_root"] if mode in ("full", "geometry") else None,
                                 image_size=cfg["image_size"], augment=False,
                                 require_depth=(mode in ("full", "geometry")))
        model = model_from_config(cfg, mode)
        model.load_state_dict(saved["model"], strict=True)
        model.to(device).eval()
        conditions = ("real", "zero_tissue", "uniform_tissue", "shuffle_tissue")
        if mode in ("full", "geometry"):
            conditions += ("flat_depth", "shuffle_depth", "shuffle_both")
        if mode == "rgb":
            conditions = ("real",)
        report, predictions = {}, {}
        for condition in conditions:
            ds = PriorAblation(base, condition, cfg["split_seed"] + idx)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            rows = infer(model, loader, device)
            predictions[condition] = rows
            report[condition] = metrics(rows)
            if condition == "real":
                report["fine_tuned_global"] = metrics(rows, "global_prediction")
            else:
                report[condition]["versus_real"] = paired(predictions["real"], rows)
                report[condition]["mean_logit_l2_change"] = float(np.mean([
                    np.linalg.norm(np.array(a["logits"]) - b["logits"])
                    for a, b in zip(predictions["real"], rows)]))
            for row in rows:
                all_rows.append({"fold": idx, "condition": condition, **row})
            print(f"fold{idx} {condition} acc={report[condition]['accuracy']:.4f} "
                  f"F1={report[condition]['macro_f1']:.4f}", flush=True)
        del model
        baseline_path = Path(cfg["rgb_checkpoint_template"].format(fold=idx))
        if sha256(baseline_path) != meta["baseline_sha256"]:
            raise ValueError("Original baseline checkpoint content changed")
        reference = model_from_config(cfg, "rgb")
        reference.load_rgb_baseline(baseline_path)
        reference.to(device).eval()
        original = infer(reference, DataLoader(base, batch_size=args.batch_size, num_workers=args.workers), device)
        report["original_rgb"] = metrics(original)
        report["paired_vs_original_rgb"] = paired(original, predictions["real"])
        report["checkpoint_sha256"] = sha256(path)
        report["baseline_sha256"] = meta["baseline_sha256"]
        report["best_epoch"] = saved["best_epoch"]
        report["n_geometry_valid"] = sum(r["geometry_valid"] for r in predictions["real"])
        pairs.append((original, predictions["real"]))
        for row in original:
            all_rows.append({"fold": idx, "condition": "original_rgb", **row})
        fold_reports[str(idx)] = report
        del reference
    aggregate = {}
    for condition in ("real", "fine_tuned_global", "original_rgb"):
        aggregate[condition] = {key: {"mean": float(np.mean([r[condition][key] for r in fold_reports.values()])),
                                     "std": float(np.std([r[condition][key] for r in fold_reports.values()]))}
                                for key in ("accuracy", "macro_f1", "mae", "qwk")}
    aggregate["paired_pooled"] = paired([r for a, _ in pairs for r in a], [r for _, b in pairs for r in b])
    if args.split == "test":
        aggregate["accuracy_delta_fold_mean_image_bootstrap_95ci"] = bootstrap_fold_mean(pairs)
    else:
        # Validation images can recur across outer folds. Do not treat these
        # repeated records as independent observations for pooled inference.
        aggregate["paired_pooled"]["mcnemar_exact_two_sided_p"] = None
        aggregate["accuracy_delta_fold_mean_image_bootstrap_95ci"] = None
        aggregate["pooled_inference_note"] = "Disabled: validation images can repeat across outer folds"
    result = {"split": args.split, "complete_three_fold": set(indices) == {0, 1, 2},
              "mode": run_identity["mode"], "train_seed": run_identity["config"]["train_seed"],
              "config": run_identity["config"], "code_fingerprint": run_identity["code"],
              "input_fingerprint": run_identity["inputs"],
              "smoke": run_identity["smoke"], "decoder": "CORN survival > 0.5 count (ordinal median)",
              "historical_rgb_accuracy_mean": 0.8168,
              "caveat": "Image-level splits and confidence intervals; patient independence not established. "
                        "A shuffled-prior test is a perturbation test, not proof of clinical validity.",
              "folds": fold_reports, "aggregate": aggregate}
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    with open(out_dir / "predictions.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        for row in all_rows:
            writer.writerow({**row, "logits": json.dumps(row["logits"])})
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
