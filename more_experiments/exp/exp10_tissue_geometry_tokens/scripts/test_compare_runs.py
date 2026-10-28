"""Synthetic comparisons reject missing controls, leakage and stale metadata."""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from compare_runs import MODES, compare_runs, metrics, write_comparison


class CompareRunsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "seed42"

    def tearDown(self):
        self.temporary.cleanup()

    def make_runs(self, split="test", folds=(0, 1, 2), repeated_val=False, rgb_matches_full=False):
        predictions = {"rgb": (0, 1, 2, 2), "tissue": (0, 1, 1, 3),
                       "geometry": (0, 0, 2, 3), "full": (0, 1, 2, 3)}
        if rgb_matches_full:
            predictions["rgb"] = predictions["full"]
        for mode in MODES:
            directory = self.root / mode / f"evaluation_{split}"
            directory.mkdir(parents=True)
            report = {"mode": mode, "split": split, "train_seed": 42,
                      "config": {"epochs": 50, "split_seed": 42},
                      "code_fingerprint": {"model.py": "fixed-code"},
                      "input_fingerprint": {"rgb": "images", "tissue": "tissues"},
                      "complete_three_fold": set(folds) == {0, 1, 2}, "smoke": False,
                      "decoder": "CORN ordinal median", "folds": {}}
            if mode in ("geometry", "full"):
                report["input_fingerprint"]["depth"] = "depths"
            records = []
            for fold in folds:
                original, real = [], []
                for target in range(4):
                    suffix = "repeat" if repeated_val and target == 0 else str(fold)
                    path = f"stage_{target+1}/image{suffix}.jpg"
                    original.append({"path": path, "target": target, "prediction": (0, 1, 1, 2)[target]})
                    real.append({"path": path, "target": target, "prediction": predictions[mode][target]})
                report["folds"][str(fold)] = {"real": metrics(real), "original_rgb": metrics(original),
                                               "baseline_sha256": f"baseline{fold}"}
                for condition, rows in (("real", real), ("original_rgb", original)):
                    records.extend({"fold": fold, "condition": condition, **row} for row in rows)
            (directory / "metrics.json").write_text(json.dumps(report))
            with open(directory / "predictions.csv", "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("fold", "condition", "path", "target", "prediction"))
                writer.writeheader()
                writer.writerows(records)

    def change_report(self, mode, change, split="test"):
        path = self.root / mode / f"evaluation_{split}" / "metrics.json"
        report = json.loads(path.read_text())
        change(report)
        path.write_text(json.dumps(report))

    def test_complete_test_pairs_separate_controls_and_checks_target(self):
        self.make_runs()
        result = compare_runs(self.root, "test")
        self.assertEqual(result["target_check"]["status"], "met_for_this_seed")
        self.assertEqual(result["training_seeds_evaluated"], 1)
        control = result["full_vs_separately_trained_controls"]["rgb"]
        self.assertEqual(control["pooled"]["fixes"], 3)
        self.assertEqual(control["pooled"]["harms"], 0)
        self.assertIsNotNone(control["pooled"]["mcnemar_exact_two_sided_p"])
        self.assertIsNotNone(control["accuracy_delta_fold_mean_image_bootstrap_95ci"])
        output = Path(self.temporary.name) / "comparison"
        write_comparison(result, output)
        self.assertTrue((output / "summary.csv").is_file())
        with self.assertRaises(FileExistsError):
            write_comparison(result, output)

    def test_missing_control_is_rejected(self):
        self.make_runs()
        shutil.rmtree(self.root / "geometry")
        with self.assertRaisesRegex(FileNotFoundError, "geometry"):
            compare_runs(self.root, "test")

    def test_mismatched_image_membership_is_rejected(self):
        self.make_runs()
        path = self.root / "tissue/evaluation_test/predictions.csv"
        path.write_text(path.read_text().replace("stage_1/image0.jpg", "stage_1/other.jpg"))
        with self.assertRaisesRegex(ValueError, "membership"):
            compare_runs(self.root, "test")

    def test_mismatched_baseline_fingerprint_is_rejected(self):
        self.make_runs()
        self.change_report("rgb", lambda report: report["folds"]["0"].update(baseline_sha256="different"))
        with self.assertRaisesRegex(ValueError, "checkpoint fingerprint"):
            compare_runs(self.root, "test")

    def test_mismatched_original_baseline_predictions_are_rejected(self):
        self.make_runs()
        path = self.root / "tissue/evaluation_test/predictions.csv"
        with open(path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row["condition"] == "original_rgb" and row["fold"] == "0" and row["target"] == "2":
                row["prediction"] = "2"
        with open(path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        altered = [{"path": row["path"], "target": int(row["target"]), "prediction": int(row["prediction"])}
                   for row in rows if row["condition"] == "original_rgb" and row["fold"] == "0"]
        self.change_report("tissue", lambda report: report["folds"]["0"].update(original_rgb=metrics(altered)))
        with self.assertRaisesRegex(ValueError, "Original RGB predictions differ"):
            compare_runs(self.root, "test")

    def test_validation_repeats_disable_inference_and_target_claim(self):
        self.make_runs("val", repeated_val=True)
        result = compare_runs(self.root, "val")
        self.assertEqual(result["prediction_records"], 12)
        self.assertEqual(result["unique_images"], 10)
        self.assertEqual(result["target_check"]["status"], "not_eligible")
        control = result["full_vs_separately_trained_controls"]["rgb"]
        self.assertIsNone(control["pooled"]["mcnemar_exact_two_sided_p"])
        self.assertIsNone(control["accuracy_delta_fold_mean_image_bootstrap_95ci"])

    def test_repeated_test_image_is_rejected(self):
        self.make_runs("test", repeated_val=True)
        with self.assertRaisesRegex(ValueError, "multiple outer folds"):
            compare_runs(self.root, "test")

    def test_partial_test_is_never_target_success(self):
        self.make_runs(folds=(0,))
        result = compare_runs(self.root, "test")
        self.assertFalse(result["complete_three_fold"])
        self.assertEqual(result["target_check"]["status"], "not_eligible")

    def test_tying_the_matched_rgb_control_is_not_target_success(self):
        self.make_runs(rgb_matches_full=True)
        result = compare_runs(self.root, "test")
        self.assertEqual(result["target_check"]["status"], "not_met")
        self.assertFalse(result["target_check"]["strictly_higher"]["separately_trained_rgb"])

    def test_smoke_and_different_training_setup_are_rejected(self):
        self.make_runs()
        self.change_report("tissue", lambda report: report.update(smoke=True))
        with self.assertRaisesRegex(ValueError, "Smoke"):
            compare_runs(self.root, "test")
        self.change_report("tissue", lambda report: report.update(smoke=False, train_seed=99))
        with self.assertRaisesRegex(ValueError, "train_seed"):
            compare_runs(self.root, "test")

    def test_altered_depth_cache_fingerprint_is_rejected(self):
        self.make_runs()
        self.change_report("geometry", lambda report: report["input_fingerprint"].update(depth="altered"))
        with self.assertRaisesRegex(ValueError, "cache fingerprints"):
            compare_runs(self.root, "test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
