"""Check interrupted training and partial-fold reports without downloads/GPU."""
import csv
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.metrics import compute_metrics  # noqa: E402


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        f"exp15_training_test_{name}", EXPERIMENT_DIR / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SyntheticDataset(Dataset):
    def __init__(self, augment):
        self.augment = augment

    def __len__(self):
        return 9

    def __getitem__(self, idx):
        values = torch.arange(8, dtype=torch.float32) / 10 + idx / 20
        if self.augment:
            # All three global RNGs affect training, as do loader shuffling
            # and the model's dropout. A seed-only resume cannot pass this.
            noise = random.random() + np.random.random() + torch.rand(()).item()
            values = values + noise / 30
        return values[:4], values[4:], idx % 3


class TinyTwoInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 12), nn.ReLU(), nn.Dropout(0.3), nn.Linear(12, 2))

    def forward(self, rgb, tissue):
        return self.net(torch.cat([rgb, tissue], dim=1))


def make_loaders(seed):
    return (
        DataLoader(SyntheticDataset(True), batch_size=3, shuffle=True, num_workers=0,
                   generator=torch.Generator().manual_seed(seed)),
        DataLoader(SyntheticDataset(False), batch_size=3, shuffle=False, num_workers=0,
                   generator=torch.Generator().manual_seed(seed + 1)),
    )


class SimulatedInterruption(RuntimeError):
    pass


class ResumeTests(unittest.TestCase):
    def test_resume_restores_rng_optimizer_schedule_and_epoch_history(self):
        engine = load_module("engine")
        config = {"epochs": 2, "lr": 0.003, "weight_decay": 0.00001, "eta_min_factor": 0.01}
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted, interrupted = root / "uninterrupted", root / "interrupted"
            uninterrupted.mkdir()
            interrupted.mkdir()

            def train(folder, model, loaders, fingerprint="same-protocol"):
                return engine.fit(
                    model, *loaders, device, config, folder / "best.pt", folder / "resume.pt",
                    fingerprint, folder / "history.csv", log_prefix="[unit-test]",
                )

            seed_all(123)
            control = TinyTwoInputModel()
            control_info = train(uninterrupted, control, make_loaders(456))
            seed_all(123)
            candidate = TinyTwoInputModel()
            original_save = engine.atomic_save

            def save_then_interrupt(value, path):
                original_save(value, path)
                if Path(path).name == "resume.pt" and value["epoch"] == 1:
                    raise SimulatedInterruption("Runtime disconnected after saving epoch one")

            with patch.object(engine, "atomic_save", save_then_interrupt):
                with self.assertRaises(SimulatedInterruption):
                    train(interrupted, candidate, make_loaders(456))
            saved = torch.load(interrupted / "resume.pt", weights_only=True)
            self.assertEqual(saved["epoch"], 1)
            self.assertEqual(len(saved["history"]), 1)

            # Simulate a fresh process with unrelated initialization and RNGs.
            seed_all(987)
            resumed = TinyTwoInputModel()
            resumed_info = train(interrupted, resumed, make_loaders(654))
            control_state = torch.load(uninterrupted / "resume.pt", weights_only=True)
            resumed_state = torch.load(interrupted / "resume.pt", weights_only=True)
            self.assertEqual(resumed_state["epoch"], 2)
            self.assertEqual(resumed_state["history"], control_state["history"])
            self.assertEqual((interrupted / "history.csv").read_text(),
                             (uninterrupted / "history.csv").read_text())
            self.assertEqual(resumed_info["best_epoch"], control_info["best_epoch"])
            self.assertEqual(resumed_info["best_val_macro_f1"], control_info["best_val_macro_f1"])
            for key in control_state["model_state"]:
                torch.testing.assert_close(control_state["model_state"][key],
                                           resumed_state["model_state"][key], rtol=0, atol=0)
            for key in control.state_dict():
                torch.testing.assert_close(control.state_dict()[key], resumed.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(control_state["scheduler_state"], resumed_state["scheduler_state"])
            self.assertEqual(control_state["optimizer_state"]["param_groups"],
                             resumed_state["optimizer_state"]["param_groups"])
            for parameter_id, parameter_state in control_state["optimizer_state"]["state"].items():
                for key, value in parameter_state.items():
                    torch.testing.assert_close(value,
                                               resumed_state["optimizer_state"]["state"][parameter_id][key],
                                               rtol=0, atol=0)

    def test_resume_rejects_a_different_protocol(self):
        engine = load_module("engine")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"protocol_fingerprint": "earlier-experiment"}, root / "resume.pt")
            with self.assertRaisesRegex(ValueError, "different experiment protocol"):
                engine.fit(
                    TinyTwoInputModel(), *make_loaders(123), torch.device("cpu"),
                    {"epochs": 2, "lr": 0.001, "weight_decay": 0.0, "eta_min_factor": 0.01},
                    root / "best.pt", root / "resume.pt", "current-experiment", root / "history.csv",
                )


def make_bundle(variant, fold, true=(0, 1, 2), predicted=(0, 1, 2)):
    stats = compute_metrics(true, predicted, num_classes=3)
    classes = ("stage_2", "stage_3", "stage_4")
    return {
        "config_id": f"mit_b1__corn__stage234__{variant}", "variant": variant,
        "encoder": "mit_b1", "head": "corn", "weights": "imagenet", "fold": fold,
        "n_test": len(true), "params_m": 1.0,
        **{key: stats[key] for key in ("accuracy", "macro_f1", "mae", "qwk")},
        "best_epoch": 1, "best_val_macro_f1": 0.5, "train_time_s": 2.0,
        "predictions": [
            {"variant": variant, "fold": fold, "path": f"{classes[y]}/fold{fold}_{idx}.npy",
             "true_index": y, "true_stage": classes[y], "pred_index": p, "pred_stage": classes[p]}
            for idx, (y, p) in enumerate(zip(true, predicted))
        ],
    }


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class ReportTests(unittest.TestCase):
    def test_unpaired_folds_do_not_generate_a_comparison(self):
        reports = load_module("reports")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundles = root / "results" / "folds"
            bundles.mkdir(parents=True)
            for variant, fold in (("rgb", 0), ("rgb_tissue", 1)):
                (bundles / f"{variant}_fold{fold}.json").write_text(json.dumps(make_bundle(variant, fold)))
            reports.write_reports(root)
            self.assertFalse((root / "results" / "paired_comparison.csv").exists())
            summary = read_csv(root / "results" / "summary.csv")
            self.assertEqual([int(row["n_folds"]) for row in summary], [1, 1])

    def test_pairing_uses_only_matching_folds_and_three_class_reports(self):
        reports = load_module("reports")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundles = root / "results" / "folds"
            bundles.mkdir(parents=True)
            inputs = [make_bundle("rgb", 0), make_bundle("rgb_tissue", 0, predicted=(0, 1, 1)),
                      make_bundle("rgb", 1, predicted=(0, 0, 0))]
            for item in inputs:
                (bundles / f"{item['variant']}_fold{item['fold']}.json").write_text(json.dumps(item))
            reports.write_reports(root)
            comparison = read_csv(root / "results" / "paired_comparison.csv")
            self.assertEqual([row["fold"] for row in comparison], ["0", "mean"])
            self.assertEqual([int(row["n_paired_folds"]) for row in comparison], [1, 1])
            expected_delta = inputs[1]["macro_f1"] - inputs[0]["macro_f1"]
            self.assertAlmostEqual(float(comparison[1]["delta_macro_f1"]), expected_delta)
            summary = {row["variant"]: row for row in read_csv(root / "results" / "summary.csv")}
            self.assertEqual(int(summary["rgb"]["n_folds"]), 2)
            self.assertEqual(int(summary["rgb_tissue"]["n_folds"]), 1)
            per_class = read_csv(root / "results" / "per_class_metrics.csv")
            self.assertEqual({row["stage"] for row in per_class}, {"stage_2", "stage_3", "stage_4"})
            pooled_support = {variant: sum(int(float(row["support"])) for row in per_class
                                          if row["variant"] == variant and row["fold"] == "pooled")
                              for variant in ("rgb", "rgb_tissue")}
            self.assertEqual(pooled_support, {"rgb": 6, "rgb_tissue": 3})
            confusions = read_csv(root / "results" / "confusion_matrices.csv")
            self.assertEqual(sum(int(row["count"]) for row in confusions
                                 if row["variant"] == "rgb_tissue" and row["fold"] == "pooled"), 3)
            self.assertEqual(len([row for row in confusions if row["variant"] == "rgb_tissue"
                                  and row["fold"] == "pooled"]), 9)


if __name__ == "__main__":
    unittest.main()
