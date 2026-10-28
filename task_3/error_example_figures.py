"""Matplotlib test figures directly from RGB, CORN values and Grad-CAM arrays.

The primary function saves a single PNG without intermediate case panels or
other figures. No model is called and input images/prediction rows stay intact.
An optional legacy helper can render errors from previously saved case panels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

IMAGE_BOXES = tuple((x, 112, x + 320, 432) for x in (38, 405, 772, 1139))
TRUE_COLOR, PRED_COLOR, OTHER_COLOR = "#23866e", "#bd4e54", "#8ba5bd"


def _relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a safe relative image path: {value}")
    return path


def _render_figure(row, crops, destination, *, save_svg=False, zero_cams=None):
    true, pred = int(row["true_label"]), int(row["pred_label"])
    fold = int(row["fold"])
    if true not in range(4) or pred not in range(4) or fold not in range(3):
        raise ValueError("Expected zero-based stage labels and fold 0..2")
    probability = np.array([row[f"prob_{c}"] for c in range(4)], dtype=float)
    survival = np.array([row[f"survival_{c}"] for c in range(3)], dtype=float)
    logits = np.array([row[f"logit_{c}"] for c in range(3)], dtype=float)
    conditional = np.array([row.get(f"conditional_{c}", .5 * (1 + np.tanh(logits[c] / 2)))
                            for c in range(3)], dtype=float)
    if (not np.isfinite(probability).all() or not np.isfinite(survival).all()
        or not np.isfinite(logits).all() or not np.isfinite(conditional).all()
        or np.any(probability < 0) or np.any((conditional < 0) | (conditional > 1))
        or not np.isclose(probability.sum(), 1., atol=1e-5)):
        raise ValueError(f"Invalid CORN values: {row['path']}")
    destination = Path(destination)
    if destination.suffix.lower() != ".png":
        raise ValueError("The test figure destination must end in .png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = int(row.get("prob_argmax", int(probability.argmax())))
    if mode not in range(4):
        raise ValueError("Probability argmax must be a zero-based stage 0..3")
    status = "CORRECT" if true == pred else "ERROR"
    decision_color = TRUE_COLOR if true == pred else PRED_COLOR

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        figure = plt.figure(figsize=(18, 10), facecolor="white")
        grid = figure.add_gridspec(2, 4, height_ratios=(1.2, 1), hspace=.43, wspace=.30,
                                  left=.055, right=.975, top=.85, bottom=.16)
        titles = ("Original RGB", f"CORN Stage {pred+1}: middle CAM",
                  f"CORN Stage {pred+1}: last CAM", f"True Stage {true+1}: last CAM")
        for index, (crop, title) in enumerate(zip(crops, titles)):
            axis = figure.add_subplot(grid[0, index])
            axis.imshow(crop)
            axis.set_title(title, fontsize=13, pad=9)
            axis.set_axis_off()
            if zero_cams is not None and zero_cams[index]:
                axis.text(.5, .035, "No positive CAM evidence", transform=axis.transAxes,
                          ha="center", va="bottom", fontsize=10, color="#51677b",
                          bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 4})

        ax_prob = figure.add_subplot(grid[1, :2])
        colors = [TRUE_COLOR if c == true else PRED_COLOR if c == pred else OTHER_COLOR for c in range(4)]
        bars = ax_prob.bar(range(4), probability, color=colors, width=.65)
        bars[pred].set_hatch("///")
        bars[pred].set_edgecolor("#6d2027")
        for c, value in enumerate(probability):
            ax_prob.text(c, value + .025, f"{value:.4f}\n({value*100:.1f}%)", ha="center", va="bottom", fontsize=11)
        ax_prob.set(xticks=range(4), xticklabels=[f"Stage {c+1}" for c in range(4)],
                    ylim=(0, 1.17), ylabel="Class probability", title="All four class probabilities")
        ax_prob.legend(handles=[Patch(facecolor=TRUE_COLOR, label=f"True stage: {true+1}"),
                                Patch(facecolor=decision_color, edgecolor="#6d2027", hatch="///",
                                      label=f"CORN prediction: {pred+1}")], loc="upper right", frameon=False)
        ax_prob.grid(axis="y", alpha=.18)
        ax_prob.set_axisbelow(True)

        ax_survival = figure.add_subplot(grid[1, 2])
        ax_survival.barh(range(3), survival, color=[TRUE_COLOR if value > .5 else OTHER_COLOR for value in survival])
        ax_survival.axvline(.5, color="#263849", ls="--", lw=1.3, label="Decision threshold 0.5")
        ax_survival.set(yticks=range(3), yticklabels=[f"> Stage {c+1}" for c in range(3)],
                        xlim=(0, 1.15), xlabel="Unconditional P(stage > k)", title="CORN survival decisions")
        ax_survival.invert_yaxis()
        for c, value in enumerate(survival):
            ax_survival.text(value + .025, c, f"{value:.3f}", va="center", fontsize=10)
        ax_survival.legend(loc="lower right", frameon=False, fontsize=9)

        ax_logits = figure.add_subplot(grid[1, 3])
        ax_logits.barh(range(3), logits, color=[OTHER_COLOR if value >= 0 else PRED_COLOR for value in logits])
        ax_logits.axvline(0, color="#263849", lw=1)
        extent = max(1., float(np.abs(logits).max()))
        ax_logits.set(yticks=range(3), yticklabels=[f"L{c}\nq={conditional[c]:.3f}" for c in range(3)],
                      xlim=(-extent*1.30, extent*1.30), xlabel="Raw conditional logit",
                      title="Three raw CORN logits")
        ax_logits.tick_params(axis="y", labelsize=9)
        ax_logits.invert_yaxis()
        for c, value in enumerate(logits):
            ax_logits.text(value + extent*.035 if value >= 0 else value - extent*.035, c,
                           f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=10)

        figure.suptitle(f"Fold {fold}  |  True stage: {true+1}  |  CORN prediction: {pred+1}  |  {status}",
                         fontsize=21, fontweight="bold", y=.975, color="#263849")
        figure.text(.5, .925, str(row["path"]), ha="center", fontsize=12, color="#51677b")
        decision = (f"P(CORN decision) = {probability[pred]:.4f}  |  "
                    f"Probability argmax = Stage {mode+1} ({probability[mode]:.4f})")
        if mode != pred:
            decision += "  |  MEDIAN AND ARGMAX DIFFER"
        figure.text(.055, .095, decision, fontsize=12, color="#263849")
        figure.text(.055, .057,
                    "CORN decision counts survival probabilities > 0.5; it is not categorical argmax. "
                    "q = sigmoid(Lk) is conditional; survival is their cumulative product.", fontsize=10, color="#51677b")
        figure.text(.055, .030,
                    "Grad-CAM: positive log P(target stage) evidence; middle 14×14 / last 7×7, each map normalized independently. "
                    "Coarse localization, not causal attention.", fontsize=10, color="#51677b")
        figure.savefig(destination, dpi=160, facecolor="white")
        if save_svg:
            figure.savefig(destination.with_suffix(".svg"), facecolor="white")
        plt.close(figure)
    return destination


def _direct_overlay(image, cam):
    values = np.asarray(cam, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Each normalized CAM must be a finite 2D array")
    if np.any(values < 0) or np.any(values > 1 + 1e-6):
        raise ValueError("CAM inputs must already be normalized to [0,1]")
    if float(values.max(initial=0)) <= 0:
        return np.array(image, copy=True), True
    field = np.asarray(Image.fromarray(values).resize(image.size, Image.Resampling.BILINEAR))
    field = np.clip(field, 0, 1)
    heat = plt.get_cmap("inferno")(field)[..., :3]
    original = np.array(image, dtype=np.float32) / 255
    alpha = (.12 + .48 * field)[..., None]
    result = np.clip(((1 - alpha) * original + alpha * heat) * 255, 0, 255).astype(np.uint8)
    return result, False


def render_test_example(image_uint8, row, cams, destination, *, save_svg=False):
    """Save one genuine Matplotlib PNG directly from an image and CAM arrays.

    ``image_uint8`` is H×W×3 RGB. ``cams`` contains normalized 2D arrays at
    ``cams['predicted']['middle']``, ``cams['predicted']['last']`` and
    ``cams['true']['last']``. The supplied row provides zero-based true/pred
    labels, fold, path, prob_0..3, survival_0..2 and logit_0..2. Both correct
    and incorrect cases are supported. SVG export is explicitly optional.
    """
    values = np.asarray(image_uint8)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError("Expected H×W×3 uint8 RGB")
    original = Image.fromarray(values).resize((320, 320), Image.Resampling.BILINEAR)
    crops, zero_cams = [np.array(original)], [False]
    for target, layer in (("predicted", "middle"), ("predicted", "last"), ("true", "last")):
        overlay, zero = _direct_overlay(original, cams[target][layer])
        crops.append(overlay)
        zero_cams.append(zero)
    return _render_figure(row, crops, destination, save_svg=save_svg, zero_cams=zero_cams)


def _render_one(row, root):
    """Compatibility path for an existing cached Pillow case panel."""
    source = root / _relative(row["case_path"])
    with Image.open(source) as panel:
        crops = [np.array(panel.convert("RGB").crop(box)) for box in IMAGE_BOXES]
    relative = Path(f"fold_{int(row['fold'])}/matplotlib_errors") / _relative(row["path"]).with_suffix(".png")
    _render_figure(row, crops, root / relative, save_svg=True)
    return relative.as_posix()


def render_error_examples(rows, run_root):
    """Save genuine Matplotlib PNG/SVG figures for all wrong test examples.

    Rows and existing PNG case panels come from ``explain_test_folds.py``.
    Return relative PNG paths; SVG files have the same stem. Input rows are
    read-only and every error is included, rather than a selected montage.
    """
    root = Path(run_root).resolve()
    result = []
    for row in rows:
        wrong = int(row["true_label"]) != int(row["pred_label"])
        if bool(row["correct"]) == wrong:
            raise ValueError(f"Inconsistent correct flag: {row['path']}")
        if wrong:
            result.append(_render_one(row, root))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.run_dir / "all_predictions.json").read_text())
    paths = render_error_examples(rows, args.run_dir)
    print(f"Saved {len(paths)} Matplotlib error figures as PNG and SVG")


if __name__ == "__main__":
    main()
