"""CPU tests for spatial behavior, safe fallbacks, training and checkpoint IO."""
import tempfile
import unittest
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from model import CornHead, TissueGeometryTokenModel, relative_depth_geometry

REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_CHECKPOINT = REPO_ROOT / "task_3/results/task_3_outputs/checkpoints/mit_b1__corn_fold0.pt"


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage3 = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.BatchNorm2d(12), nn.GELU())
        self.patch_embed4 = nn.Conv2d(12, 16, 3, padding=1)
        self.block4 = nn.Sequential(nn.GELU(), nn.Dropout2d(0.2))
        self.norm4 = nn.BatchNorm2d(16)

    def forward(self, rgb):
        stage3 = self.stage3(F.avg_pool2d(rgb, 2))
        stage4 = self.norm4(self.block4(self.patch_embed4(F.avg_pool2d(stage3, 2))))
        return [stage3, stage4]


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.out_dim = 16
        self.stage3_dim = 12

    def forward(self, rgb):
        return self.encoder(rgb)[-1].mean(dim=(2, 3))


def inputs(batch=2):
    rgb = torch.randn(batch, 3, 24, 24)
    tissue = torch.zeros(batch, 5, 24, 24)
    tissue[:, 0, 5:19, 5:12] = 0.8
    tissue[:, 1, 5:19, 5:12] = 0.2
    tissue[:, 2, 5:19, 12:19] = 0.7
    tissue[:, 3, 5:19, 12:19] = 0.3
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 24), torch.linspace(-1, 1, 24), indexing="ij")
    depth = (xx.square() + yy.square() + 0.3 * xx)[None, None].repeat(batch, 1, 1, 1)
    return rgb, tissue, depth, torch.ones(batch, dtype=torch.bool)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(32)
        self.model = TissueGeometryTokenModel(backbone=TinyBackbone(), token_dim=16, dropout=0)
        self.batch = inputs()

    def test_empty_wound_is_exact_finite_rgb_fallback(self):
        self.model.eval()
        rgb, tissue, depth, flags = self.batch
        tissue[0].zero_()
        output = self.model(rgb, tissue, depth, flags)
        self.assertTrue(torch.equal(output["logits"][0], output["rgb_logits"][0]))
        self.assertTrue(torch.equal(output["local_logits"][0], torch.zeros(3)))
        self.assertFalse(output["has_wound"][0])
        self.assertFalse(output["geometry_valid"][0])
        self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_relative_depth_and_predictions_are_affine_invariant(self):
        self.model.eval()
        rgb, tissue, depth, flags = self.batch
        support = tissue.sum(1, keepdim=True)
        relative, valid = relative_depth_geometry(depth, support, flags)
        transformed, valid2 = relative_depth_geometry(3.7 * depth + 11.5, support, flags)
        torch.testing.assert_close(relative, transformed, atol=5e-6, rtol=5e-6)
        self.assertTrue(torch.equal(valid, valid2))
        first = self.model(rgb, tissue, depth, flags)
        second = self.model(rgb, tissue, 3.7 * depth + 11.5, flags)
        torch.testing.assert_close(first["logits"], second["logits"], atol=1e-6, rtol=1e-6)

    def test_invalid_and_missing_geometry_match_tissue_mode(self):
        self.model.eval()
        rgb, tissue, depth, flags = self.batch
        for variant in (None, torch.ones_like(depth), torch.full_like(depth, float("nan"))):
            full = self.model(rgb, tissue, variant, flags)
            self.model.mode = "tissue"
            tissue_only = self.model(rgb, tissue, depth, flags)
            self.model.mode = "full"
            self.assertFalse(full["geometry_valid"].any())
            torch.testing.assert_close(full["logits"], tissue_only["logits"], atol=0, rtol=0)
        disabled = self.model(rgb, tissue, depth, torch.zeros_like(flags))
        missing = self.model(rgb, tissue, None, flags)
        torch.testing.assert_close(disabled["logits"], missing["logits"], atol=0, rtol=0)

    def test_tissue_arrangement_changes_local_predictions(self):
        self.model.eval()
        rgb, tissue, depth, flags = self.batch
        first = self.model(rgb, tissue, depth, flags)
        # Same per-class mass and wound support, different spatial association.
        swapped = tissue.flip(-1)
        second = self.model(rgb, swapped, depth, flags)
        self.assertTrue(torch.equal(tissue.sum((2, 3)), swapped.sum((2, 3))))
        self.assertGreater((first["local_logits"] - second["local_logits"]).abs().max().item(), 1e-6)
        self.assertTrue(torch.equal(first["rgb_logits"], second["rgb_logits"]))

    def test_geometry_changes_local_predictions(self):
        self.model.eval()
        rgb, tissue, depth, flags = self.batch
        first = self.model(rgb, tissue, depth, flags)
        second = self.model(rgb, tissue, depth.flip(-1), flags)
        self.assertTrue(first["geometry_valid"].all())
        self.assertGreater((first["local_logits"] - second["local_logits"]).abs().max().item(), 1e-6)

    def test_first_step_gradients_reach_spatial_branch_and_last_stage(self):
        self.model.set_last_stage_trainable(True)
        self.model.train()
        output = self.model(*self.batch)
        labels = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
        loss = F.binary_cross_entropy_with_logits(output["logits"], labels)
        loss = loss + 0.25 * F.binary_cross_entropy_with_logits(output["local_logits"], labels)
        loss.backward()
        for name in ("spatial_projection.0.weight", "token_projection.0.weight",
                     "query_attention.in_proj_weight", "ordinal_queries",
                     "tissue_embeddings", "bin_embeddings", "local_head.1.weight",
                     "rgb_backbone.encoder.patch_embed4.weight"):
            gradient = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        self.assertIsNone(self.model.rgb_backbone.encoder.stage3[0].weight.grad)

    def test_only_last_stage_is_unfrozen_and_frozen_stages_stay_eval(self):
        self.model.set_last_stage_trainable(True)
        self.model.train()
        encoder = self.model.rgb_backbone.encoder
        self.assertFalse(encoder.stage3.training)
        self.assertTrue(encoder.block4.training)
        self.assertFalse(encoder.stage3[0].weight.requires_grad)
        self.assertTrue(encoder.patch_embed4.weight.requires_grad)
        self.model.eval()
        self.assertFalse(encoder.block4.training)
        self.model.set_last_stage_trainable(False)
        self.assertFalse(any(p.requires_grad for p in self.model.rgb_backbone.parameters()))

    def test_sequential_checkpoint_load_is_exact_and_strict(self):
        baseline = nn.Sequential(TinyBackbone(), CornHead(16)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.pt"
            torch.save(baseline.state_dict(), path)
            self.model.load_rgb_baseline(path).eval()
            output = self.model(*self.batch)
            self.assertTrue(torch.equal(output["rgb_logits"], baseline(self.batch[0])))
            state = baseline.state_dict()
            state.pop("1.fc.bias")
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                self.model.load_rgb_baseline(path)
            state["other.weight"] = torch.ones(1)
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                self.model.load_rgb_baseline(path)

    def test_rgb_mode_returns_original_pathway(self):
        self.model.mode = "rgb"
        self.model.eval()
        output = self.model(*self.batch)
        self.assertTrue(torch.equal(output["logits"], output["rgb_logits"]))
        self.assertFalse(output["geometry_valid"].any())

    def test_geometry_control_removes_tissue_composition(self):
        self.model.mode = "geometry"
        self.model.eval()
        rgb, tissue, depth, flags = self.batch
        first = self.model(rgb, tissue, depth, flags)
        reassigned = tissue.roll(2, dims=1)
        second = self.model(rgb, reassigned, depth, flags)
        torch.testing.assert_close(first["logits"], second["logits"], atol=0, rtol=0)
        self.assertTrue(first["geometry_valid"].all())

    def test_misaligned_tissue_shape_is_rejected(self):
        rgb, tissue, depth, flags = self.batch
        with self.assertRaisesRegex(ValueError, "spatial coordinates"):
            self.model(rgb, tissue[..., :-1, :], depth, flags)

    @unittest.skipUnless(REAL_CHECKPOINT.is_file(), "real fold0 baseline is not available locally")
    def test_real_mit_baseline_path_matches_original_task3_exactly(self):
        spec = importlib.util.spec_from_file_location("original_task3_model", REPO_ROOT / "task_3/src/model.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        baseline = original.build_model("mit_b1", None, "corn").eval()
        baseline.load_state_dict(torch.load(REAL_CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
        model = TissueGeometryTokenModel(weights=None, dropout=0).load_rgb_baseline(REAL_CHECKPOINT).eval()
        rgb = torch.randn(1, 3, 224, 224)
        tissue = torch.zeros(1, 5, 224, 224)
        with torch.no_grad():
            baseline_logits = baseline(rgb)
            output = model(rgb, tissue)
        self.assertTrue(torch.equal(output["rgb_logits"], baseline_logits))
        self.assertTrue(torch.equal(output["logits"], baseline_logits))


if __name__ == "__main__":
    unittest.main(verbosity=2)
