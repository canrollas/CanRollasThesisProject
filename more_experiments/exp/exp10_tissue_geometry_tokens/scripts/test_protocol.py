"""Tests for the research protocol, independent of claimed model accuracy."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from evaluate import PriorAblation, paired
from train import input_fingerprint, resolve_config, validate_baseline_protocol, EXP_DIR


class FakeDataset:
    def __len__(self):
        return 17

    def __getitem__(self, idx):
        return {"rgb": torch.full((3, 2, 2), float(idx)),
                "tissue": torch.full((5, 2, 2), float(idx)),
                "depth": torch.full((1, 2, 2), float(idx)),
                "depth_valid": torch.tensor(True), "path": str(idx), "target": idx % 4}


class ProtocolTests(unittest.TestCase):
    def test_shuffle_is_global_derangement_and_keeps_rgb_target(self):
        base = FakeDataset()
        changed = PriorAblation(base, "shuffle_both", 42)
        again = PriorAblation(base, "shuffle_both", 42)
        for i in range(len(base)):
            self.assertNotEqual(i, changed.donors[i])
            self.assertEqual(changed.donors[i], again.donors[i])
            self.assertTrue(torch.equal(changed[i]["rgb"], base[i]["rgb"]))
            self.assertEqual(changed[i]["target"], base[i]["target"])
            self.assertTrue(torch.equal(changed[i]["depth"], base[int(changed.donors[i])]["depth"]))

    def test_input_identity_detects_changed_cache_and_survives_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {key: str(root / key) for key in ("data_root", "tissue_root", "depth_root")}
            samples = [("stage_1/example.jpg", 0)]
            for key in cfg:
                p = Path(cfg[key]) / "stage_1" / ("example.jpg" if key == "data_root" else "example.npy")
                p.parent.mkdir(parents=True)
                p.write_bytes(key.encode())
            original = input_fingerprint(cfg, samples, "full")
            moved = copy.deepcopy(cfg)
            import shutil
            for key in cfg:
                moved[key] = str(root / "moved" / key)
                shutil.copytree(cfg[key], moved[key])
            self.assertEqual(original, input_fingerprint(moved, samples, "full"))
            (Path(cfg["depth_root"]) / "stage_1/example.npy").write_bytes(b"different valid cache")
            self.assertNotEqual(original, input_fingerprint(cfg, samples, "full"))

    def test_historical_checkpoint_rejects_changed_split_seed_before_training(self):
        cfg = resolve_config(EXP_DIR / "configs/base.yaml")
        cfg["split_seed"] = 43
        with self.assertRaisesRegex(ValueError, "split_seed"):
            validate_baseline_protocol(cfg, [])

    def test_paired_comparison_counts_fixes_harms_and_checks_order(self):
        original = [{"path": str(i), "target": 0, "prediction": p} for i, p in enumerate([0, 1, 1, 0])]
        proposed = [{"path": str(i), "target": 0, "prediction": p} for i, p in enumerate([1, 0, 0, 0])]
        result = paired(original, proposed)
        self.assertEqual((result["fixes"], result["harms"], result["net_correct"]), (2, 1, 1))
        self.assertEqual(result["prediction_flips"], 3)
        with self.assertRaises(ValueError):
            paired(original, list(reversed(proposed)))


if __name__ == "__main__":
    unittest.main()
