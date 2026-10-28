"""Readable per-image CORN explanations and held-out scientific figures.

These functions never run or alter a model. Per-image heat maps are normalized
separately and show positive Grad-CAM evidence, not causal attention weights.
The confidence assigned to a CORN prediction is P(the ordinal-median class),
which need not equal the largest categorical probability.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps, font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (average_precision_score, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

STAGES = ("Stage 1", "Stage 2", "Stage 3", "Stage 4")
COLORS = ("#3677a6", "#2f9d86", "#d7a141", "#c96163")
BG, INK, MUTED = "#f7f9fc", "#16283b", "#566b80"
PANEL_SIZE = (1500, 900)
IMAGE_BOXES = tuple((x, 112, x + 320, 432) for x in (38, 405, 772, 1139))


@lru_cache(maxsize=12)
def _font(size):
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def _short(text, limit=95):
    return str(text) if len(str(text)) <= limit else str(text)[:limit - 3] + "..."


def _camera(cams, target, layer):
    group = cams.get(target, {})
    value = group.get(layer, group.get("mid") if layer == "middle" else None)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"CAM {target}/{layer} must be a finite 2D map")
    return np.clip(array, 0, 1)


def _overlay(image, cam):
    if cam is None or float(np.max(cam, initial=0)) <= 0:
        return image.copy(), False
    resized = np.array(Image.fromarray(cam).resize(image.size, Image.Resampling.BILINEAR))
    heat = colormaps["inferno"](resized)[..., :3]
    rgb = np.array(image, dtype=np.float32) / 255
    # Make zero evidence mostly transparent, rather than painting every pixel.
    alpha = (.12 + .48 * resized)[..., None]
    mixed = np.clip(((1 - alpha) * rgb + alpha * heat) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(mixed), True


def _text(draw, xy, text, size=19, color=INK):
    draw.text(xy, str(text), fill=color, font=_font(size))


def _probability_bars(draw, row, origin, width):
    x, y = origin
    _text(draw, (x, y), "Categorical stage probabilities", 22)
    true, pred = int(row["true_label"]), int(row["pred_label"])
    for c in range(4):
        yy = y + 45 + c * 53
        value = float(row[f"prob_{c}"])
        suffix = ("  TRUE" if c == true else "") + ("  CORN" if c == pred else "")
        _text(draw, (x, yy), f"S{c + 1}{suffix}", 16)
        left, right = x + 140, x + width - 60
        draw.rounded_rectangle((left, yy, right, yy + 23), 4, fill="#e5eaf1")
        if value > 0:
            draw.rounded_rectangle((left, yy, left + (right - left) * value, yy + 23), 4, fill=COLORS[c])
        _text(draw, (right + 9, yy), f"{value:.3f}", 16)


def _survival_bars(draw, row, origin, width):
    x, y = origin
    _text(draw, (x, y), "CORN decisions: P(stage > k)", 22)
    left, right = x + 85, x + width - 60
    for c in range(3):
        yy, value = y + 45 + c * 53, float(row[f"survival_{c}"])
        _text(draw, (x, yy), f"> S{c + 1}", 18)
        draw.rounded_rectangle((left, yy, right, yy + 23), 4, fill="#e5eaf1")
        if value > 0:
            draw.rounded_rectangle((left, yy, left + (right - left) * value, yy + 23), 4,
                                   fill="#2f9d86" if value > .5 else "#8298ae")
        cut = (left + right) / 2
        draw.line((cut, yy - 4, cut, yy + 28), fill=INK, width=2)
        _text(draw, (right + 9, yy), f"{value:.3f}", 16)
    _text(draw, (x, y + 220), "Each survival > 0.5 crosses one stage.", 16, MUTED)
    _text(draw, (x, y + 246), "This decoder is not categorical argmax.", 16, MUTED)


def _logit_bars(draw, row, origin, width):
    x, y = origin
    _text(draw, (x, y), "Raw conditional CORN logits", 22)
    values = [float(row[f"logit_{c}"]) for c in range(3)]
    extent = max(max(abs(v) for v in values), 1.)
    left, right = x + 52, x + width - 65
    center = (left + right) / 2
    for c, value in enumerate(values):
        yy = y + 45 + c * 53
        _text(draw, (x, yy), f"L{c}", 18)
        draw.rectangle((left, yy + 2, right, yy + 23), fill="#e5eaf1")
        stop = center + value / extent * (right - left) / 2
        draw.rectangle((min(center, stop), yy + 2, max(center, stop), yy + 23),
                       fill="#3677a6" if value >= 0 else "#c96163")
        draw.line((center, yy - 4, center, yy + 28), fill=INK, width=2)
        _text(draw, (right + 9, yy), f"{value:+.2f}", 16)
    _text(draw, (x, y + 220), "sigmoid(Lk) is conditional, not class P.", 16, MUTED)
    _text(draw, (x, y + 246), "Survival = cumulative product of sigmoids.", 16, MUTED)


def render_case(image_uint8, row, cams, destination):
    """Save one compact PNG with RGB/CAMs and actual CORN decision values.

    ``cams[target][layer]`` contains normalized 2D maps; target is predicted,
    true, or threshold_0..2 and layer is middle/last. Labels are zero based.
    """
    rgb = np.asarray(image_uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected H×W×3 uint8 RGB")
    true, pred = int(row["true_label"]), int(row["pred_label"])
    if true not in range(4) or pred not in range(4):
        raise ValueError("Expected zero-based stages 0..3")
    canvas = Image.new("RGB", PANEL_SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    status = "CORRECT" if true == pred else "ERROR"
    _text(draw, (38, 20), f"Fold {row['fold']}  |  True: S{true + 1}  |  CORN prediction: S{pred + 1}  |  {status}", 25)
    _text(draw, (38, 58), _short(row["path"], 105), 17, MUTED)
    original = Image.fromarray(rgb).resize((320, 320), Image.Resampling.BILINEAR)
    titles = ("Original RGB", "Predicted stage: middle layer", "Predicted stage: last layer", "True stage: last layer")
    maps = (None, _camera(cams, "predicted", "middle"), _camera(cams, "predicted", "last"),
            _camera(cams, "true", "last"))
    for index, (box, title, cam) in enumerate(zip(IMAGE_BOXES, titles, maps)):
        _text(draw, (box[0], 87), title, 17)
        overlay, positive = (original.copy(), True) if index == 0 else _overlay(original, cam)
        canvas.paste(overlay, box)
        if not positive:
            draw.rectangle((box[0] + 5, box[1] + 275, box[2] - 5, box[3] - 5), fill="#f1f3f7")
            _text(draw, (box[0] + 12, box[1] + 287), "No positive CAM evidence", 16, MUTED)
    strip = np.repeat(colormaps["inferno"](np.linspace(0, 1, 280))[None, :, :3], 12, axis=0)
    canvas.paste(Image.fromarray(np.uint8(strip * 255)), (1140, 450))
    _text(draw, (1140, 466), "0      per-map normalization      1", 15, MUTED)
    _text(draw, (38, 450), "Positive Grad-CAM of log P(target stage); coarse spatial evidence, not causal attention.", 17, MUTED)
    _text(draw, (38, 480), "Middle: 14 x 14  |  Last: 7 x 7; each map independently scaled to [0, 1].", 16, MUTED)
    _logit_bars(draw, row, (38, 535), 445)
    _survival_bars(draw, row, (530, 535), 445)
    _probability_bars(draw, row, (1022, 535), 440)
    assigned = float(row.get("predicted_probability", row[f"prob_{pred}"]))
    entropy = float(row.get("entropy", -sum(float(row[f"prob_{c}"]) * np.log(max(float(row[f"prob_{c}"]), 1e-30)) for c in range(4))))
    mode = int(row.get("prob_argmax", np.argmax([row[f"prob_{c}"] for c in range(4)])))
    _text(draw, (38, 840), f"P(CORN stage)={assigned:.3f}  |  Entropy={entropy:.3f} nats  |  Probability mode=S{mode + 1}", 19)
    if mode != pred:
        _text(draw, (910, 840), "CORN median and probability mode differ.", 17, "#a45327")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, compress_level=3)
    return destination


def render_threshold_case(image_uint8, row, cams, destination):
    """Show what supports each raw conditional ordinal-boundary logit.

    The target is Lk, not log class probability. Positive evidence for a
    conditional boundary differs from evidence for the final decoded stage.
    """
    rgb = np.asarray(image_uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected H×W×3 uint8 RGB")
    canvas = Image.new("RGB", PANEL_SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    _text(draw, (38, 20), f"Fold {row['fold']}  |  True: S{int(row['true_label'])+1}  |  CORN: S{int(row['pred_label'])+1}  |  Ordinal boundaries", 24)
    _text(draw, (38, 58), _short(row['path'], 105), 17, MUTED)
    original = Image.fromarray(rgb)
    canvas.paste(original.resize((360, 360), Image.Resampling.BILINEAR), (38, 160))
    _text(draw, (38, 130), "Original RGB", 20)
    _text(draw, (38, 555), "Targets are raw logits L0, L1, L2.", 19)
    for k, text in enumerate(("L0: stage > 1", "L1: stage > 2 given stage > 1", "L2: stage > 3 given stage > 2")):
        _text(draw, (38, 597 + k*37), text, 17, MUTED)
    _text(draw, (38, 735), "A boundary can have no positive", 18, MUTED)
    _text(draw, (38, 764), "Grad-CAM after the ReLU step.", 18, MUTED)
    small = original.resize((300, 300), Image.Resampling.BILINEAR)
    for k, x in enumerate((455, 800, 1145)):
        _text(draw, (x, 88), f"L{k}={float(row[f'logit_{k}']):+.3f}", 22)
        _text(draw, (x, 118), f"Conditional P={float(row[f'conditional_{k}']):.3f}", 18, MUTED)
        for layer, y in (("middle", 165), ("last", 510)):
            _text(draw, (x, y-25), f"{layer.title()} layer", 17)
            overlay, positive = _overlay(small, _camera(cams, f"threshold_{k}", layer))
            canvas.paste(overlay, (x, y))
            if not positive:
                draw.rectangle((x+4, y+256, x+296, y+295), fill="#f1f3f7")
                _text(draw, (x+9, y+268), "No positive CAM evidence", 16, MUTED)
    _text(draw, (38, 848), "Positive Grad-CAM of raw conditional logits; each map independently normalized. This is not causal attention.", 18, MUTED)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, compress_level=3)
    return destination


def calibration_bins(probability, correct, n_bins=10):
    """Equal-width empirical calibration of the actual selected-class P."""
    probability, correct = np.asarray(probability, float), np.asarray(correct, bool)
    if probability.shape != correct.shape or not len(probability) or not np.isfinite(probability).all():
        raise ValueError("Expected nonempty aligned finite confidence/correct arrays")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must lie in [0,1]")
    membership = np.minimum((probability * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.
    for idx in range(n_bins):
        mask = membership == idx
        count = int(mask.sum())
        mean_conf = float(probability[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count / len(probability) * abs(accuracy - mean_conf)
        bins.append({"lower": idx / n_bins, "upper": (idx + 1) / n_bins,
                     "count": count, "confidence": mean_conf, "accuracy": accuracy})
    return {"ece": float(ece), "n_bins": n_bins, "bins": bins}


def _save(fig, directory, stem, figures):
    fig.savefig(directory / f"{stem}.png", dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures.extend([f"{stem}.png", f"{stem}.svg"])


def _metrics(y, pred):
    return {"n": len(y), "accuracy": float(np.mean(y == pred)),
            "macro_f1": float(f1_score(y, pred, labels=range(4), average="macro", zero_division=0)),
            "mae": float(np.abs(y - pred).mean()),
            "qwk": float(cohen_kappa_score(y, pred, labels=range(4), weights="quadratic")),
            "confusion": confusion_matrix(y, pred, labels=range(4)).tolist()}


def _gallery(rows, directory, name, title):
    selected = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        available = [row for row in rows if int(row["fold"]) == fold]
        selected.extend(sorted(available, key=lambda row: float(row["predicted_probability"]), reverse=True)[:4])
    if not selected:
        return None
    columns, tile_w, tile_h = 3, 800, 330
    gallery = Image.new("RGB", (columns * tile_w, 90 + int(np.ceil(len(selected) / columns)) * tile_h), BG)
    draw = ImageDraw.Draw(gallery)
    _text(draw, (28, 20), title, 28)
    _text(draw, (28, 58), "Up to 4 highest P(CORN stage) errors per fold; each tile: original | predicted last-layer CAM.", 19, MUTED)
    for index, row in enumerate(selected):
        x, y = index % columns * tile_w, 90 + index // columns * tile_h
        path = Path(row.get("case_path", ""))
        candidates = [path] if path.is_absolute() else [directory.parent / path, directory / path]
        found = next((candidate for candidate in candidates if candidate.is_file()), None)
        if found:
            with Image.open(found) as panel:
                original = panel.crop(IMAGE_BOXES[0]).resize((280, 280), Image.Resampling.BILINEAR)
                cam = panel.crop(IMAGE_BOXES[2]).resize((280, 280), Image.Resampling.BILINEAR)
                gallery.paste(original, (x + 40, y + 35))
                gallery.paste(cam, (x + 340, y + 35))
        _text(draw, (x + 30, y + 4), f"F{row['fold']} S{int(row['true_label']) + 1} -> S{int(row['pred_label']) + 1}  P={float(row['predicted_probability']):.3f}", 21)
        _text(draw, (x + 635, y + 65), _short(Path(row['path']).stem, 12), 16, MUTED)
    path = directory / f"{name}.png"
    gallery.save(path, compress_level=3)
    return path.name


def aggregate_figures(rows, outdir):
    """Create held-out plots from one row per original test image.

    ``outdir`` should be RUN_ROOT/summary; case_path is relative to RUN_ROOT.
    Returns JSON-compatible statistics and a list of generated filenames.
    """
    rows, outdir = list(rows), Path(outdir)
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Expected one unique out-of-fold row per source image")
    outdir.mkdir(parents=True, exist_ok=True)
    y = np.array([int(row["true_label"]) for row in rows])
    pred = np.array([int(row["pred_label"]) for row in rows])
    fold = np.array([int(row["fold"]) for row in rows])
    p = np.array([[float(row[f"prob_{c}"]) for c in range(4)] for row in rows])
    logits = np.array([[float(row[f"logit_{c}"]) for c in range(3)] for row in rows])
    survival = np.array([[float(row[f"survival_{c}"]) for c in range(3)] for row in rows])
    if (not np.isfinite(p).all() or not np.isfinite(logits).all() or not np.isfinite(survival).all()
        or np.any((y < 0) | (y > 3) | (pred < 0) | (pred > 3))
        or np.any((p < -1e-7) | (p > 1 + 1e-7)) or not np.allclose(p.sum(1), 1, atol=1e-5)):
        raise ValueError("Invalid class probabilities, labels or logits")
    if not np.array_equal(pred, (survival > .5).sum(1)):
        raise ValueError("Saved predictions must use the historical CORN ordinal-median decoder")
    if not np.allclose(np.cumsum(p[:, ::-1], axis=1)[:, ::-1][:, 1:], survival, atol=2e-5):
        raise ValueError("Saved categorical probabilities and survival outputs disagree")
    correct, mode = y == pred, p.argmax(1)
    assigned, maximum = p[np.arange(len(rows)), pred], p.max(1)
    entropy = -(p * np.log(np.maximum(p, 1e-30))).sum(1)
    figures, stats = [], {"pooled": _metrics(y, pred), "folds": {}}
    for idx in sorted(set(fold)):
        stats["folds"][str(idx)] = _metrics(y[fold == idx], pred[fold == idx])
    stats.update({"decoder": "CORN ordinal median: count(survival > 0.5)",
                  "probability_mode_disagreements": int((mode != pred).sum()),
                  "probability_mode_accuracy": float((mode == y).mean()),
                  "decoded_class_probability_mean": float(assigned.mean()),
                  "entropy_mean_nats": float(entropy.mean()),
                  "categorical_log_loss": float(log_loss(y, p, labels=range(4))),
                  "multiclass_brier_score": float(np.square(p - np.eye(4)[y]).sum(1).mean()),
                  "decoded_calibration": calibration_bins(assigned, correct),
                  "argmax_calibration": calibration_bins(maximum, mode == y),
                  "stage3_to_stage4": int(((y == 2) & (pred == 3)).sum()),
                  "stage4_to_stage3": int(((y == 3) & (pred == 2)).sum())})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    subsets = [(f"Fold {idx}", fold == idx) for idx in sorted(set(fold))] + [("Pooled out-of-fold", np.ones(len(y), bool))]
    for normalize, stem in ((False, "confusion_counts"), (True, "confusion_row_normalized")):
        fig, axes = plt.subplots(1, len(subsets), figsize=(4.1 * len(subsets), 4.2), constrained_layout=True, squeeze=False)
        for ax, (name, mask) in zip(axes[0], subsets):
            counts = confusion_matrix(y[mask], pred[mask], labels=range(4))
            values = counts / np.maximum(counts.sum(1, keepdims=True), 1) if normalize else counts
            ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else max(1, int(counts.max())))
            for a in range(4):
                for b in range(4):
                    ax.text(b, a, f"{values[a,b]:.1%}" if normalize else str(counts[a,b]), ha="center", va="center",
                            color="white" if values[a,b] > (.5 if normalize else counts.max() / 2) else INK)
            ax.set(xticks=range(4), yticks=range(4), xticklabels=STAGES, yticklabels=STAGES,
                   xlabel="CORN prediction", ylabel="True stage", title=f"{name}\nN={int(mask.sum())}; accuracy={correct[mask].mean():.3f}")
        _save(fig, outdir, stem, figures)
    for values, label, stem, cut in ((logits, "Conditional logit", "logits_by_true_stage", 0),
                                    (survival, "Unconditional survival probability", "survival_by_true_stage", .5)):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
        for c, ax in enumerate(axes):
            groups = [values[y == stage, c] for stage in range(4)]
            boxes = ax.boxplot(groups, patch_artist=True, showfliers=False)
            for patch, color in zip(boxes['boxes'], COLORS):
                patch.set_facecolor(color); patch.set_alpha(.65)
            for stage in range(4):
                errors = values[(y == stage) & ~correct, c]
                jitter = np.random.default_rng(100 + c * 10 + stage).uniform(-.13, .13, len(errors))
                ax.scatter(stage + 1 + jitter, errors, color="#a9343e", s=9, alpha=.4, label="Misclassified" if stage == 0 else None)
            ax.axhline(cut, color="#3c4c5d", ls="--", lw=1)
            ax.set(xticks=range(1, 5), xticklabels=STAGES, ylabel=label,
                   title=f"{'L' + str(c) if stem.startswith('logits') else 'P(stage > S' + str(c + 1) + ')'}")
            if stem.startswith("survival"): ax.set_ylim(-.03, 1.03)
        axes[0].legend(loc="best", fontsize=9)
        fig.suptitle("Distribution by true stage; red points are held-out errors", fontsize=13)
        _save(fig, outdir, stem, figures)
    order = np.lexsort((-assigned, pred, y))
    fig, axes = plt.subplots(3, 1, figsize=(16, 8.7), sharex=True, constrained_layout=True)
    for ax, values, title, cmap, minimum, maximum_value in (
        (axes[0], p, "Categorical class probabilities", "viridis", 0, 1),
        (axes[1], survival, "CORN survival probabilities (decision threshold 0.5)", "viridis", 0, 1),
        (axes[2], logits, "Raw conditional logits (different meaning from class scores)", "coolwarm", -max(1, np.abs(logits).max()), max(1, np.abs(logits).max()))):
        image = ax.imshow(values[order].T, aspect="auto", interpolation="nearest", cmap=cmap, vmin=minimum, vmax=maximum_value)
        ax.set_title(title); ax.set_yticks(range(values.shape[1]))
        ax.set_yticklabels(STAGES if values.shape[1] == 4 else [f"> S{i+1}" for i in range(3)])
        for boundary in np.cumsum(np.bincount(y, minlength=4))[:-1]: ax.axvline(boundary - .5, color="white", lw=1)
        fig.colorbar(image, ax=ax, fraction=.02, pad=.01)
    axes[-1].set_xlabel("One held-out source image per column; sorted by true stage, prediction and assigned probability")
    _save(fig, outdir, "all_test_outputs_heatmap", figures)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for mask, label, color in ((correct, "Correct", "#2f9d86"), (~correct, "Error", "#c96163")):
        axes[0,0].hist(assigned[mask], bins=np.linspace(0, 1, 21), alpha=.6, color=color, label=label)
        axes[0,1].hist(entropy[mask], bins=np.linspace(0, np.log(4), 21), alpha=.6, color=color, label=label)
    axes[0,0].set(xlabel="P(CORN decoded stage)", ylabel="Images", title="Assigned probability and errors")
    axes[0,1].set(xlabel="Categorical entropy (nats)", ylabel="Images", title="Uncertainty and errors")
    for ax in axes[0]: ax.legend()
    ranked = np.argsort(-assigned, kind="stable")
    axes[1,0].plot(np.arange(1,len(y)+1)/len(y), np.cumsum(correct[ranked])/np.arange(1,len(y)+1), color="#3677a6")
    axes[1,0].axhline(correct.mean(), color="#555", ls="--", label="All images")
    axes[1,0].set(xlabel="Retained coverage, highest assigned P first", ylabel="Accuracy", title="Descriptive confidence / coverage", ylim=(0,1.02)); axes[1,0].legend()
    disagreement = mode != pred
    axes[1,1].scatter(assigned[~disagreement], maximum[~disagreement], s=10, alpha=.3, color="#8298ae", label="Median = mode")
    axes[1,1].scatter(assigned[disagreement], maximum[disagreement], s=24, alpha=.7, color="#c96163", label=f"Median != mode (n={disagreement.sum()})")
    axes[1,1].plot([0,1],[0,1],color="#555",ls="--")
    axes[1,1].set(xlabel="P(CORN decoded stage)", ylabel="Largest class probability", title="Decoder confidence is not always max P", xlim=(0,1), ylim=(0,1)); axes[1,1].legend(fontsize=8)
    _save(fig, outdir, "confidence_uncertainty_errors", figures)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), constrained_layout=True)
    for key, label, color in (("decoded_calibration", "CORN decoded stage", "#3677a6"), ("argmax_calibration", "Categorical mode", "#c96163")):
        result = stats[key]; bins = [b for b in result["bins"] if b["count"]]
        axes[0].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], 'o-', color=color, label=f"{label}; ECE={result['ece']:.3f}")
    axes[0].plot([0,1],[0,1],color="#555",ls="--"); axes[0].legend(fontsize=9)
    axes[0].set(xlabel="Mean selected-class probability", ylabel="Observed accuracy", title="Uncalibrated out-of-fold reliability", xlim=(0,1), ylim=(0,1))
    bins = stats["decoded_calibration"]["bins"]
    axes[1].bar([(b["lower"]+b["upper"])/2 for b in bins], [b["count"] for b in bins], width=.085, color="#3677a6")
    axes[1].set(xlabel="P(CORN decoded stage)", ylabel="Images", title="Equal-width calibration bin counts", xlim=(0,1))
    _save(fig, outdir, "probability_calibration", figures)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    stats["one_vs_rest"] = {}
    for c, color in enumerate(COLORS):
        binary = y == c
        if len(np.unique(binary)) < 2: continue
        fpr, tpr, _ = roc_curve(binary, p[:,c]); precision, recall, _ = precision_recall_curve(binary,p[:,c])
        auc, ap = float(roc_auc_score(binary,p[:,c])), float(average_precision_score(binary,p[:,c]))
        axes[0].plot(fpr,tpr,color=color,label=f"S{c+1}, AUC={auc:.3f}")
        axes[1].plot(recall,precision,color=color,label=f"S{c+1}, AP={ap:.3f}")
        stats["one_vs_rest"][str(c)] = {"roc_auc":auc,"average_precision":ap}
    axes[0].plot([0,1],[0,1],color="#777",ls="--")
    for ax,title,xlabel,ylabel in ((axes[0],"OOF class-probability ROC","False-positive rate","True-positive rate"), (axes[1],"OOF class-probability precision / recall","Recall","Precision")):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel,xlim=(0,1),ylim=(0,1.02)); ax.legend(fontsize=9)
    _save(fig, outdir, "class_probability_roc_pr", figures)
    # Region comparisons are descriptive against model-predicted masks.
    # Undefined mass (zero CAM) and enrichment (absent region) are omitted.
    if all(f"cam_{region}_mass" in row for row in rows for region in ("wound", "skin", "background")):
        def values_for(key):
            return np.array([float(row[key]) if row.get(key) is not None else np.nan for row in rows])
        def finite_mean(values):
            finite = np.asarray(values)[np.isfinite(values)]
            return float(finite.mean()) if finite.size else None
        stats["positive_cam_location"] = {"interpretation": "descriptive overlap with predicted skin/wound masks; enrichment=positive CAM mass / region area; undefined omitted"}
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        for index, region in enumerate(("wound", "skin", "background")):
            area, mass = values_for(f"{region}_area_fraction"), values_for(f"cam_{region}_mass")
            enrichment, middle_enrichment = values_for(f"cam_{region}_enrichment"), values_for(f"middle_cam_{region}_enrichment")
            for mask, label, color in ((correct,"Correct","#2f9d86"),(~correct,"Error","#c96163")):
                available = mask & np.isfinite(area) & np.isfinite(mass)
                axes[0,index].scatter(area[available],mass[available],s=11,alpha=.35,color=color,label=label)
            axes[0,index].plot([0,1],[0,1],ls="--",color="#777",label="Uniform spatial mass")
            axes[0,index].set(title=f"{region.title()}: positive CAM mass",xlabel="Region image-area fraction",ylabel="Last-layer positive CAM mass",xlim=(0,1),ylim=(0,1))
            for layer, values, offset, color in (("Last",enrichment,-.15,"#3677a6"),("Middle",middle_enrichment,.15,"#d7a141")):
                groups = [values[(y==stage)&np.isfinite(values)] for stage in range(4)]
                # Boxplot accepts empty groups, but uses a NaN placeholder for
                # missing regions instead of pretending their enrichment=0.
                groups = [group if len(group) else np.array([np.nan]) for group in groups]
                boxes=axes[1,index].boxplot(groups,positions=np.arange(1,5)+offset,widths=.24,
                                          patch_artist=True,showfliers=False,manage_ticks=False)
                for box in boxes['boxes']: box.set_facecolor(color); box.set_alpha(.7)
                axes[1,index].plot([],[],color=color,lw=6,label=layer)
            axes[1,index].axhline(1,ls="--",color="#777")
            axes[1,index].set(xticks=range(1,5),xticklabels=STAGES,title=f"{region.title()}: area-adjusted enrichment",ylabel="CAM mass / image-area fraction (symlog)")
            axes[1,index].set_yscale("symlog",linthresh=.25)
            axes[1,index].legend(fontsize=9)
            finite=enrichment[np.isfinite(enrichment)]
            stats["positive_cam_location"][region]={
                "n_defined_enrichment":int(len(finite)),"mean_area_fraction":finite_mean(area),
                "mean_cam_mass":finite_mean(mass),"mean_enrichment":finite_mean(enrichment),
                "median_enrichment":float(np.median(finite)) if len(finite) else None,
                "correct_mean_enrichment":finite_mean(enrichment[correct]),
                "error_mean_enrichment":finite_mean(enrichment[~correct]),
                "middle_mean_enrichment":finite_mean(middle_enrichment)}
        axes[0,0].legend(fontsize=8)
        fig.suptitle("Positive Grad-CAM overlap with predicted region masks; enrichment > 1 exceeds uniform spatial mass",fontsize=13)
        _save(fig,outdir,"cam_region_mass_and_enrichment",figures)
    error_rows = [row for row in rows if int(row["true_label"]) != int(row["pred_label"])]
    for selected,name,title in ((error_rows,"highest_confidence_errors","Highest-confidence held-out errors"),
        ([r for r in error_rows if int(r["true_label"])==2 and int(r["pred_label"])==3],"stage3_to4_errors","Stage 3 -> Stage 4 errors"),
        ([r for r in error_rows if int(r["true_label"])==3 and int(r["pred_label"])==2],"stage4_to3_errors","Stage 4 -> Stage 3 errors")):
        gallery = _gallery(selected,outdir,name,title)
        if gallery: figures.append(gallery)
    return {"figures":figures,"stats":stats}
