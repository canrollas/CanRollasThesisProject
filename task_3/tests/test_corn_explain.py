"""Probability/decoder and gradient-evidence regression tests."""
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corn_explain import corn_outputs, extract_explanations


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(3, 2, 1)
        self.second = nn.Conv2d(2, 2, 1)

    def forward(self, x):
        middle = torch.relu(self.first(x))
        last = self.second(torch.nn.functional.avg_pool2d(middle, 2))
        return [x, x[:, :0], middle, middle, middle, last]


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()

    def forward(self, x):
        return self.encoder(x)[-1].mean((-2, -1))


def make_model():
    torch.manual_seed(8)
    return nn.Sequential(TinyBackbone(), nn.Linear(2, 3)).eval()


class CornExplainTests(unittest.TestCase):
    def test_probabilities_match_independent_survival_differences(self):
        torch.manual_seed(1)
        logits = torch.randn(64, 3, dtype=torch.float64) * 6
        out = corn_outputs(logits)
        s = logits.sigmoid().cumprod(1)
        expected = torch.cat((1 - s[:, :1], s[:, :-1] - s[:, 1:], s[:, -1:]), 1)
        torch.testing.assert_close(out["class_probabilities"], expected, atol=1e-14, rtol=1e-13)
        torch.testing.assert_close(out["class_probabilities"].sum(1), torch.ones(64, dtype=torch.float64))
        self.assertTrue((out["class_probabilities"] >= 0).all())

    def test_decoding_matches_historical_corn_and_can_differ_from_argmax(self):
        from coral_pytorch.dataset import corn_label_from_logits
        logits = torch.logit(torch.tensor([[.6, .9, .9], [.5, .5, .5], [.9, .9, .9]]))
        out = corn_outputs(logits)
        torch.testing.assert_close(out["decoded"], corn_label_from_logits(logits))
        self.assertEqual(int(out["decoded"][0]), 2)
        self.assertEqual(int(out["probability_argmax"][0]), 3)
        self.assertEqual(int(out["decoded"][1]), 0)

    def test_extreme_logits_have_finite_log_targets_and_gradients(self):
        logits = torch.tensor([[1000., -1000., 1000.], [-1000., 1000., -1000.]], requires_grad=True)
        out = corn_outputs(logits)
        self.assertTrue(torch.isfinite(out["log_class_probabilities"]).all())
        self.assertTrue(torch.isfinite(out["class_probabilities"]).all())
        out["log_class_probabilities"].sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_binary_case_and_invalid_inputs(self):
        out = corn_outputs(torch.zeros(2, 1))
        torch.testing.assert_close(out["class_probabilities"], torch.full((2, 2), .5))
        for bad in (torch.zeros(3), torch.zeros(2, 0), torch.ones(2, 3, dtype=torch.int64),
                    torch.tensor([[float("nan"), 1., 2.]])):
            with self.assertRaises(ValueError):
                corn_outputs(bad)

    def test_extract_returns_all_targets_layers_and_valid_normalization(self):
        model = make_model()
        result = extract_explanations(model, torch.randn(3, 3, 8, 8), torch.tensor([0, 1, 3]))
        self.assertEqual(set(result["cams"]), {"predicted", "true", "threshold_0", "threshold_1", "threshold_2"})
        for maps in result["cams"].values():
            self.assertEqual(maps["last"]["raw"].shape, (3, 4, 4))
            self.assertEqual(maps["middle"]["raw"].shape, (3, 8, 8))
            for layer in maps.values():
                self.assertTrue(torch.isfinite(layer["normalized"]).all())
                self.assertTrue((layer["normalized"] >= 0).all())
                self.assertTrue((layer["normalized"] <= 1).all())
                self.assertFalse(layer["normalized"].requires_grad)
                torch.testing.assert_close(layer["raw"], layer["signed"].relu())

    def test_class_cam_targets_actual_median_decoder(self):
        model = make_model()
        with torch.no_grad():
            model[1].weight.zero_()
            model[1].bias.copy_(torch.logit(torch.tensor([.6, .9, .9])))
        out = extract_explanations(model, torch.randn(1, 3, 8, 8), include_thresholds=False)
        self.assertEqual(int(out["target_classes"]["predicted"][0]), 2)
        self.assertEqual(int(out["probability_argmax"][0]), 3)
        self.assertTrue(out["cams"]["predicted"]["last"]["zero"].all())

    def test_threshold_cam_agrees_with_linear_gap_analytic_gradient(self):
        model = make_model()
        image = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            activation = model[0].encoder(image)[-1]
            expected = (activation * (model[1].weight[0] / 16)[None, :, None, None]).sum(1)
        out = extract_explanations(model, image)
        torch.testing.assert_close(out["cams"]["threshold_0"]["last"]["signed"], expected)

    def test_batched_explanations_match_individual_images(self):
        model = make_model()
        images = torch.randn(3, 3, 8, 8)
        batch = extract_explanations(model, images, [1, 2, 0])
        single = [extract_explanations(model, images[i:i + 1], [label])
                  for i, label in enumerate((1, 2, 0))]
        for target in batch["cams"]:
            for layer in ("last", "middle"):
                expected = torch.cat([s["cams"][target][layer]["raw"] for s in single])
                torch.testing.assert_close(batch["cams"][target][layer]["raw"], expected, atol=1e-8, rtol=1e-5)

    def test_model_parameters_grads_and_input_are_unchanged(self):
        model = make_model()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = copy.deepcopy(model.state_dict())
        grads = [p.grad.clone() for p in model.parameters()]
        image = torch.randn(2, 3, 8, 8)
        original = image.clone()
        with torch.no_grad():
            extract_explanations(model, image, [1, 2])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        for parameter, gradient in zip(model.parameters(), grads):
            torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)
        torch.testing.assert_close(image, original, atol=0, rtol=0)
        self.assertFalse(model.training)
        self.assertFalse(model[0].encoder._forward_hooks)

    def test_failed_call_removes_hook_and_rejects_training_mode(self):
        model = make_model()
        with self.assertRaises(ValueError):
            extract_explanations(model, torch.randn(2, 3, 8, 8), [0, 6])
        self.assertFalse(model[0].encoder._forward_hooks)
        with self.assertRaises(ValueError):
            extract_explanations(model.train(), torch.randn(2, 3, 8, 8))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is optional")
    def test_cuda_outputs_remain_device_correct(self):
        logits = torch.randn(4, 3, device="cuda", requires_grad=True)
        out = corn_outputs(logits)
        self.assertEqual(out["class_probabilities"].device.type, "cuda")
        out["log_class_probabilities"].sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
