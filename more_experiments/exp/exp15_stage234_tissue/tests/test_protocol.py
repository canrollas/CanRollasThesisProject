"""Small protocol checks; no pretrained downloads or clinical images required.

Run from the repository root:
    python -m unittest discover -s more_experiments/exp/exp15_stage234_tissue/tests
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "task_3"))
sys.path.insert(0, str(EXPERIMENT_DIR / "scripts"))

from src.losses import build_loss  # noqa: E402
from src.metrics import compute_metrics, predict_labels  # noqa: E402
from src.splits import build_folds  # noqa: E402


def load_experiment_module(name):
    spec = importlib.util.spec_from_file_location(
        f"exp15_test_{name}", EXPERIMENT_DIR / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FoldProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.counts = (21, 24, 27, 30)
        for stage, count in enumerate(self.counts, start=1):
            folder = self.root / f"stage_{stage}"
            folder.mkdir()
            for idx in range(count):
                np.save(folder / f"image_{idx:03d}.npy", np.zeros((8, 8, 8), np.uint8))
        self.data = load_experiment_module("data")

    def test_filtering_preserves_original_train_val_and_test_memberships(self):
        original = build_folds(self.root, num_folds=3, val_fraction=0.25, seed=42)
        actual = self.data.build_stage234_folds(
            self.root, num_folds=3, val_fraction=0.25, seed=42
        )
        self.assertEqual(len(actual), 3)
        for source_fold, filtered_fold in zip(original, actual):
            for split in ("train", "val", "test"):
                expected = [(path, label - 1) for path, label in source_fold[split] if label > 0]
                self.assertEqual(filtered_fold[split], expected, split)

    def test_stage1_excluded_labels_remapped_and_test_coverage_exactly_once(self):
        folds = self.data.build_stage234_folds(self.root, val_fraction=0.25)
        all_test_paths = []
        for fold in folds:
            split_paths = {}
            for split in ("train", "val", "test"):
                split_paths[split] = {path for path, _ in fold[split]}
                for path, label in fold[split]:
                    self.assertFalse(path.startswith("stage_1/"))
                    self.assertIn(label, (0, 1, 2))
                    self.assertEqual(label, int(Path(path).parts[0].split("_")[1]) - 2)
            self.assertTrue(split_paths["train"].isdisjoint(split_paths["val"]))
            self.assertTrue(split_paths["train"].isdisjoint(split_paths["test"]))
            self.assertTrue(split_paths["val"].isdisjoint(split_paths["test"]))
            all_test_paths.extend(path for path, _ in fold["test"])
        self.assertEqual(len(all_test_paths), sum(self.counts[1:]))
        self.assertEqual(len(all_test_paths), len(set(all_test_paths)))


class PairedInputTests(unittest.TestCase):
    def test_rgb_values_and_joint_spatial_transform_preserve_tissue_alignment(self):
        data = load_experiment_module("data")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stage_3").mkdir()
            arr = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8).astype(np.uint8)
            arr[:, :, 3] = arr[:, :, 0]
            np.save(root / "stage_3" / "sample.npy", arr)

            def transform(image):
                return np.ascontiguousarray(np.rot90(image[:, ::-1], 1))

            dataset = data.PairedStageDataset(
                [("stage_3/sample.npy", 1)], root, image_size=8, transform=transform
            )
            rgb, tissue, label = dataset[0]
            expected = transform(arr)
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            actual_rgb = ((rgb * std + mean) * 255.0).numpy().transpose(1, 2, 0)
            actual_tissue = (tissue * 255.0).numpy().transpose(1, 2, 0)
            np.testing.assert_allclose(actual_rgb, expected[:, :, :3], atol=3e-5)
            np.testing.assert_allclose(actual_tissue, expected[:, :, 3:], atol=3e-5)
            np.testing.assert_allclose(actual_rgb[:, :, 0], actual_tissue[:, :, 0], atol=3e-5)
            self.assertEqual(tuple(rgb.shape), (3, 8, 8))
            self.assertEqual(tuple(tissue.shape), (5, 8, 8))
            self.assertEqual(label, 1)


class ThreeClassCornTests(unittest.TestCase):
    def test_three_class_loss_decoder_and_macro_f1(self):
        logits = torch.tensor([[-8.0, 8.0], [8.0, -8.0], [8.0, 8.0]], requires_grad=True)
        targets = torch.tensor([0, 1, 2])
        predictions = predict_labels(logits, "corn")
        self.assertEqual(predictions.tolist(), targets.tolist())
        loss = build_loss("corn", 3)(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        metrics = compute_metrics(targets.numpy(), predictions.numpy(), num_classes=3)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(tuple(metrics["confusion"].shape), (3, 3))


class TinyBackbone(nn.Module):
    """Keeps the real CORN/fusion heads while avoiding a model download."""

    def __init__(self, encoder_name, weights):
        super().__init__()
        self.out_dim = 8
        self.conv = nn.Conv2d(3, self.out_dim, kernel_size=1)

    def forward(self, images):
        return self.conv(images).mean(dim=(2, 3))


class ModelBranchTests(unittest.TestCase):
    def test_both_heads_have_two_logits_and_tissue_path_only_affects_fusion(self):
        module = load_experiment_module("model")
        rgb_input = torch.randn(2, 3, 16, 16)
        tissue_input = torch.rand(2, 5, 16, 16, requires_grad=True)
        with patch.object(module, "RGBBackbone", TinyBackbone):
            torch.manual_seed(42)
            rgb = module.build_variant("rgb", weights=None).eval()
            torch.manual_seed(42)
            fusion = module.build_variant("rgb_tissue", weights=None).eval()
        for key, value in rgb.rgb_backbone.state_dict().items():
            torch.testing.assert_close(value, fusion.rgb_backbone.state_dict()[key])
        rgb_output = rgb(rgb_input, tissue_input)
        self.assertEqual(tuple(rgb_output.shape), (2, 2))
        torch.testing.assert_close(rgb_output, rgb(rgb_input, torch.zeros_like(tissue_input)))
        fused_output = fusion(rgb_input, tissue_input)
        self.assertEqual(tuple(fused_output.shape), (2, 2))
        fused_output.sum().backward()
        self.assertIsNotNone(tissue_input.grad)
        self.assertGreater(tissue_input.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(fused_output).all().item())


if __name__ == "__main__":
    unittest.main()
