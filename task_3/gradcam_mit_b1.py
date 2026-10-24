#!/usr/bin/env python3
"""Grad-CAM (Selvaraju et al., 2017) for the mit_b1__corn stage classifier.

MiT-B1 is a hierarchical transformer, but segmentation_models_pytorch's
encoder wrapper (the same one src/model.py's Backbone calls) returns a
list of per-stage feature maps in plain [B, C, H, W] layout -- exactly the
shape Grad-CAM was designed for in CNNs, so no transformer-specific
adaptation is needed: we hook the encoder's output list directly.

For each sample this saves a 3-panel PNG (original | Grad-CAM heatmap |
overlay) for two target layers:
  - stage 5 (last stage, [B,512,7,7])  -- same feature map exp-7 pooled
    into F_global; most semantically abstracted, lowest resolution.
  - stage 3 ([B,320,14,14]) -- same feature map exp-7's ROI-align read
    from for F_local; higher spatial resolution, more localized.

The CORN head has no single "class logit" the way a softmax head does, so
the backprop target is the decoded probability of the predicted class
(corn_logits_to_class_probs(logits)[0, pred_class]) -- a well-defined
scalar in [0, 1] that increases exactly when the model becomes more
confident in that specific stage.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import cm
from PIL import Image

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR))

from src.dataset import CLASS_ORDER, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.model import build_model  # noqa: E402

IMAGE_SIZE = 224
# SMP's mit_b1 encoder returns a 6-element list: [0] input passthrough,
# [1] an empty (0-channel) placeholder, then the 4 real hierarchical
# stages at strides 4/8/16/32 -- indices 2,3,4,5 with 64/128/320/512
# channels respectively. index 5 is the [B,512,7,7] map exp-7 pooled into
# F_global; index 4 is the [B,320,14,14] map exp-7's ROI-align read as
# F_local. Both require_grad=True; index 1 does not (verified directly) --
# never hook that one.
STAGE_TARGETS = {"stage5_last": 5, "stage3_mid": 4}


def corn_logits_to_class_probs(logits):
    cond = torch.sigmoid(logits)
    survival = torch.cumprod(cond, dim=1)
    ones = torch.ones(survival.shape[0], 1, dtype=survival.dtype)
    zeros = torch.zeros(survival.shape[0], 1, dtype=survival.dtype)
    s_prev = torch.cat([ones, survival], dim=1)
    s_next = torch.cat([survival, zeros], dim=1)
    probs = (s_prev - s_next).clamp(min=0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)


def load_image_tensor(data_root, rel_path):
    image = Image.open(Path(data_root) / rel_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image_np = np.array(image, dtype=np.uint8)
    image_f = image_np.astype(np.float32) / 255.0
    normed = (image_f - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, image_np


class GradCAM:
    """Hooks the encoder's forward call once; each run() call re-reads
    whichever stage index it's asked for from the same captured list, so
    one forward+backward pass yields both target layers' CAMs at once."""

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = {}
        self.handle = model[0].encoder.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.activations = out  # list of per-stage feature maps, still in the graph
        for idx in STAGE_TARGETS.values():
            out[idx].retain_grad()

    def run(self, image_tensor, target_class=None):
        self.model.zero_grad()
        image_tensor = image_tensor.clone().requires_grad_(True)
        logits = self.model(image_tensor)
        probs = corn_logits_to_class_probs(logits)
        pred_class = probs.argmax(dim=1).item()
        score_class = target_class if target_class is not None else pred_class
        probs[0, score_class].backward()

        cams = {}
        for name, stage_idx in STAGE_TARGETS.items():
            act = self.activations[stage_idx][0].detach()  # [C,H,W]
            grad = self.activations[stage_idx].grad[0]  # [C,H,W]
            weights = grad.mean(dim=(1, 2))  # [C], standard Grad-CAM GAP-over-gradients
            cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
            cam = cam / cam.max().clamp(min=1e-12)
            cams[name] = cam.numpy()
        return pred_class, probs[0].detach().numpy(), cams

    def close(self):
        self.handle.remove()


def overlay_heatmap(image_np, cam, alpha=0.45):
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    cam_resized = np.array(cam_img, dtype=np.float32) / 255.0
    heat_rgb = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heat_rgb + (1 - alpha) * image_np).astype(np.uint8)
    return heat_rgb, overlay


def save_panel(image_np, heat_rgb, overlay, out_path):
    panel = np.concatenate([image_np, heat_rgb, overlay], axis=1)
    Image.fromarray(panel).save(out_path)


def pick_samples(data_root, fold_ckpt_samples, per_class=2):
    """per_class correctly-classified + per_class misclassified samples,
    one of each where available, spread across all 4 stages."""
    picked = []
    by_class = {c: [] for c in range(len(CLASS_ORDER))}
    for rel_path, label in fold_ckpt_samples:
        by_class[label].append(rel_path)
    for label, rels in by_class.items():
        picked.extend((r, label) for r in rels[:per_class])
    return picked


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0, help="which fold's checkpoint/test split to use")
    p.add_argument("--per-class", type=int, default=2, help="how many test images per stage to visualize")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = TASK_DIR.parent
    data_root = repo_root / "datasets" / "wound-stage-classification"
    checkpoint_path = TASK_DIR / "results" / "task_3_outputs" / "checkpoints" / f"mit_b1__corn_fold{args.fold}.pt"
    out_dir = Path(args.out_dir) if args.out_dir else TASK_DIR / "results" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(TASK_DIR))
    from src.splits import build_folds  # noqa: E402
    from src.utils import load_yaml  # noqa: E402
    base_cfg = load_yaml(TASK_DIR / "configs" / "base.yaml")
    folds = build_folds(data_root, base_cfg["num_folds"], base_cfg["val_fraction"], base_cfg["seed"])
    test_samples = folds[args.fold]["test"]

    model = build_model("mit_b1", None, "corn", num_classes=len(CLASS_ORDER))
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    gradcam = GradCAM(model)
    samples = pick_samples(data_root, test_samples, per_class=args.per_class)

    print(f"fold={args.fold} visualizing {len(samples)} samples -> {out_dir}")
    for rel_path, true_label in samples:
        image_tensor, image_np = load_image_tensor(data_root, rel_path)
        pred_class, probs, cams = gradcam.run(image_tensor)

        stem = Path(rel_path).stem
        true_name, pred_name = CLASS_ORDER[true_label], CLASS_ORDER[pred_class]
        tag = "correct" if true_label == pred_class else "WRONG"

        for layer_name, cam in cams.items():
            heat_rgb, overlay = overlay_heatmap(image_np, cam)
            out_path = out_dir / f"{stem}__true-{true_name}__pred-{pred_name}__{tag}__{layer_name}.png"
            save_panel(image_np, heat_rgb, overlay, out_path)

        print(f"  {stem:28s} true={true_name:8s} pred={pred_name:8s} ({tag}) "
              f"probs={np.round(probs, 3).tolist()}")

    gradcam.close()
    print(f"done -- panels saved under {out_dir}")


if __name__ == "__main__":
    main()
