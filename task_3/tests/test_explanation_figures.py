"""Numerical/visual artifact checks using synthetic held-out records."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explanation_figures import (PANEL_SIZE, aggregate_figures,
                                 calibration_bins, render_case, render_threshold_case)


def record(index, true, pred, probabilities):
    p = np.asarray(probabilities, float)
    survival = np.array((p[1:].sum(), p[2:].sum(), p[3]))
    cond = survival / np.r_[1., survival[:-1]]
    logits = np.log(cond / (1 - cond))
    row = {"path": f"stage_{true+1}/case{index}.jpg", "fold": index % 3,
           "true_label": true, "pred_label": pred, "prob_argmax": int(p.argmax()),
           "correct": true == pred, "predicted_probability": float(p[pred]),
           "case_path": f"cases/case{index}.png"}
    for c in range(4): row[f"prob_{c}"] = float(p[c])
    for c in range(3):
        row[f"logit_{c}"] = float(logits[c])
        row[f"conditional_{c}"] = float(cond[c])
        row[f"survival_{c}"] = float(survival[c])
    return row


class ExplanationFigureTests(unittest.TestCase):
    def test_selected_class_calibration_does_not_substitute_max_probability(self):
        # Ordinal median is class1 although class0 has the largest probability.
        result = calibration_bins([.35], [True])
        self.assertAlmostEqual(result["ece"], .65)
        self.assertEqual(sum(b["count"] for b in result["bins"]), 1)
        self.assertAlmostEqual(calibration_bins([0., 1.], [False, True])["ece"], 0.)
        with self.assertRaises(ValueError): calibration_bins([1.2], [True])

    def test_per_case_panel_preserves_original_array_and_accepts_zero_cam(self):
        with tempfile.TemporaryDirectory() as folder:
            rgb = np.full((224, 224, 3), (40, 90, 150), dtype=np.uint8)
            original = rgb.copy()
            row = record(0, 0, 1, [.4, .35, .2, .05])
            cam = np.zeros((7, 7), dtype=np.float32)
            path = render_case(rgb, row, {"predicted": {"last": cam, "middle": cam},
                                         "true": {"last": cam}}, Path(folder) / "case.png")
            with Image.open(path) as image:
                self.assertEqual(image.size, PANEL_SIZE)
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((70, 180)), (40, 90, 150))
            np.testing.assert_array_equal(rgb, original)

    def test_conditional_logit_panel_saves_six_maps_without_mutating_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            rgb = np.full((224, 224, 3), (70, 120, 150), dtype=np.uint8)
            row = record(0, 2, 3, [.02, .03, .35, .6])
            cam = np.eye(7, dtype=np.float32)
            cams = {f"threshold_{k}": {"last": cam, "middle": cam} for k in range(3)}
            path = render_threshold_case(rgb, row, cams, Path(folder) / "thresholds.png")
            with Image.open(path) as image:
                self.assertEqual(image.size, PANEL_SIZE)
                self.assertEqual(image.getpixel((50, 170)), (70, 120, 150))
            np.testing.assert_array_equal(cam, np.eye(7, dtype=np.float32))

    def test_aggregate_outputs_and_decoder_specific_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [record(0, 0, 1, [.4, .35, .2, .05]),
                    record(1, 1, 1, [.05, .8, .1, .05]),
                    record(2, 2, 3, [.05, .05, .35, .55]),
                    record(3, 3, 2, [.05, .1, .6, .25]),
                    record(4, 0, 0, [.97, .01, .01, .01]),
                    record(5, 1, 1, [.05, .8, .1, .05]),
                    record(6, 2, 2, [.05, .05, .8, .1]),
                    record(7, 3, 3, [.02, .03, .05, .9])]
            yy, xx = np.mgrid[:14, :14]
            cam = np.exp(-((xx-7)**2+(yy-7)**2)/15).astype(np.float32)
            for row in rows:
                for region, area, mass in (("wound",.25,.5),("skin",.6,.4),("background",.15,.1)):
                    row[f"{region}_area_fraction"] = area
                    row[f"cam_{region}_mass"] = mass
                    row[f"cam_{region}_enrichment"] = mass / area
                    row[f"middle_cam_{region}_mass"] = mass
                    row[f"middle_cam_{region}_enrichment"] = mass / area
                render_case(np.full((224, 224, 3), (90, 110, 70), np.uint8), row,
                            {"predicted": {"middle": cam, "last": cam}, "true": {"last": cam}},
                            root / row["case_path"])
            report = aggregate_figures(rows, root / "summary")
            self.assertEqual(report["stats"]["pooled"]["n"], 8)
            self.assertEqual(report["stats"]["probability_mode_disagreements"], 1)
            self.assertEqual(report["stats"]["stage3_to_stage4"], 1)
            self.assertEqual(report["stats"]["stage4_to_stage3"], 1)
            self.assertAlmostEqual(report["stats"]["pooled"]["accuracy"], 5/8)
            self.assertAlmostEqual(report["stats"]["positive_cam_location"]["wound"]["mean_enrichment"], 2.)
            for name in report["figures"]:
                self.assertGreater((root / "summary" / name).stat().st_size, 100)
            self.assertIn("highest_confidence_errors.png", report["figures"])
            json.dumps(report, allow_nan=False)
            with self.assertRaises(ValueError): aggregate_figures([rows[0], rows[0]], root / "duplicate")


if __name__ == "__main__":
    unittest.main()
