"""Semantic data checks: baseline parity, spatial alignment and corrupt priors."""
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import IMAGENET_MEAN, IMAGENET_STD, StagePriorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "baseline_augmentation", REPO_ROOT / "task_3/src/augmentations.py"
)
baseline_augmentation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_augmentation)


class PriorDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for folder in ("rgb", "tissue", "depth"):
            (self.root / folder / "stage_1").mkdir(parents=True)
        self.relative_path = "stage_1/example.png"
        self.rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        Image.fromarray(self.rgb).save(self.root / "rgb" / self.relative_path)
        self.cache = np.concatenate((self.rgb, np.repeat(self.rgb[:, :, :1], 5, axis=2)), axis=2)
        self.tissue_path = self.root / "tissue/stage_1/example.npy"
        self.depth_path = self.root / "depth/stage_1/example.npy"
        np.save(self.tissue_path, self.cache)
        np.save(self.depth_path, self.rgb[:, :, 0].astype(np.float32))

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, **kwargs):
        return StagePriorDataset([(self.relative_path, 0)], self.root / "rgb",
                                 self.root / "tissue", self.root / "depth",
                                 image_size=8, **kwargs)

    def test_rgb_matches_baseline_and_priors_remain_aligned(self):
        dataset = self.dataset(augment=True, require_depth=True)
        for seed in range(16):
            random.seed(seed)
            expected = baseline_augmentation.ImageAugmentation()(self.rgb)
            random.seed(seed)
            item = dataset[0]
            expected_rgb = ((expected.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD)
            np.testing.assert_allclose(item["rgb"].numpy(), expected_rgb.transpose(2, 0, 1), atol=1e-6)
            np.testing.assert_allclose(item["tissue"][0].numpy() * 255, expected[:, :, 0], atol=2e-5)
            np.testing.assert_array_equal(item["depth"][0].numpy(), expected[:, :, 0])
            self.assertTrue(item["depth_valid"].item())
            self.assertEqual(item["depth_valid"].dtype, torch.bool)
            self.assertEqual(item["depth_valid"].ndim, 0)

    def test_missing_depth_has_explicit_invalid_flag(self):
        self.depth_path.unlink()
        item = self.dataset()[0]
        self.assertFalse(item["depth_valid"].item())
        self.assertEqual(item["depth"].count_nonzero().item(), 0)
        with self.assertRaises(FileNotFoundError):
            self.dataset(require_depth=True)

    def test_corrupt_depth_cannot_silently_become_missing(self):
        for invalid in (np.full((8, 8), np.nan, dtype=np.float32),
                        np.ones((8, 8), dtype=np.float32),
                        np.arange(64, dtype=np.uint8).reshape(8, 8),
                        np.arange(64, dtype=np.float32).reshape(4, 16)):
            np.save(self.depth_path, invalid)
            with self.assertRaises(ValueError):
                self.dataset(require_depth=False)[0]

    def test_tissue_cache_must_correspond_to_original_image(self):
        self.cache[0, 0, 0] += 1
        np.save(self.tissue_path, self.cache)
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            self.dataset()[0]

    def test_resizing_preserves_types_and_shapes(self):
        item = StagePriorDataset([(self.relative_path, 0)], self.root / "rgb",
                                 self.root / "tissue", self.root / "depth",
                                 image_size=16, require_depth=True)[0]
        for key, channels in (("rgb", 3), ("tissue", 5), ("depth", 1)):
            self.assertEqual(item[key].shape, (channels, 16, 16))
            self.assertEqual(item[key].dtype, torch.float32)
            self.assertTrue(torch.isfinite(item[key]).all())


if __name__ == "__main__":
    unittest.main()
