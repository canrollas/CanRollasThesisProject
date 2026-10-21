"""Experiment 7 (Chapter 8): global-local ROI feature fusion, Steps 3-5.

Adapts the global-local fusion architecture from Bayraktar et al. (2026,
Fig. 4) -- NOT the attention-enhanced two-stream model (Fig. 5): no
attention mechanism anywhere in this file.

Reuses task_3/src/model.py's own Backbone/CornHead classes directly (same
pattern as exp5/6's model.py) so a baseline checkpoint's state-dict keys
("0." for Backbone, "1." for CornHead) load into the exact same submodule
types -- no reimplementation, no key-name drift.

MiT-B1 encoder(x) returns 6 feature maps for a 224x224 input (verified
empirically, not assumed):
  [B,3,224,224] [B,0,112,112] [B,64,56,56] [B,128,28,28] [B,320,14,14] [B,512,7,7]
F_global source = feats[-1] = [B,512,7,7] (same as the baseline's own GAP).
F_local source   = feats[-2] = [B,320,14,14] (the "third stage", higher
spatial resolution than the final stage) -> roi_align spatial_scale =
14/224 = 0.0625.

Baseline-preserving init (Step 5): rather than a separate residual branch
(exp5/6's approach), this widens the CornHead's own Linear(512,3) to
Linear(640,3) directly -- the spec's own suggested "widened input
projection". The 512 global columns + bias are copied EXACTLY from the
baseline checkpoint; the 128 new local columns are zero-initialized. At
construction this makes fusion_logits == baseline_logits EXACTLY
(torch.equal, verified in test_model.py), since the local branch's
contribution to the logits is zero regardless of local_proj's own
(ordinarily-initialized) weights -- only the HEAD's local columns need
zeroing, not local_proj itself.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.ops import roi_align

THIS_DIR = Path(__file__).resolve().parent  # exp/exp7_global_local_roi_fusion/scripts/
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import Backbone as RGBBackbone, CornHead  # noqa: E402

GLOBAL_DIM = 512   # feats[-1] channels, mit_b1
LOCAL_IN_CHANNELS = 320  # feats[-2] channels, mit_b1 ("third stage")
STAGE3_STRIDE = 16  # 224 / 14, verified empirically above


class MultiStageBackbone(nn.Module):
    """Wraps task_3's own Backbone (so checkpoint keys line up exactly for
    loading) but exposes the raw multi-stage feature list instead of only
    the globally-pooled final-stage vector."""

    def __init__(self, encoder_name="mit_b1", weights=None):
        super().__init__()
        self.inner = RGBBackbone(encoder_name, weights)  # has .encoder, .out_dim, .kind
        assert self.inner.kind == "smp"
        assert self.inner.out_dim == GLOBAL_DIM

    def forward(self, x):
        feats = self.inner.encoder(x)
        feat_stage3 = feats[-2]  # [B, 320, 14, 14]
        feat_final = feats[-1]   # [B, 512, 7, 7]
        return feat_stage3, feat_final


class LocalROIProjection(nn.Module):
    """ROIAlign output [B,C,k,k] -> spatial-average-pool -> Linear projection
    to a compact embedding. Ordinary (non-zero) init -- see module
    docstring for why this doesn't break baseline-reproduction at init."""

    def __init__(self, in_channels=LOCAL_IN_CHANNELS, out_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_dim)

    def forward(self, roi_feat):
        pooled = roi_feat.mean(dim=(2, 3))  # [B, in_channels]
        return self.proj(pooled)  # [B, out_dim]


class GlobalLocalFusionHead(nn.Module):
    """Single widened CornHead-equivalent: Linear(global_dim+local_dim,
    num_classes-1). init_from_baseline() does the zero-init described in
    the module docstring."""

    def __init__(self, global_dim=GLOBAL_DIM, local_dim=128, num_classes=4):
        super().__init__()
        self.global_dim = global_dim
        self.fc = nn.Linear(global_dim + local_dim, num_classes - 1)

    def forward(self, fused):
        return self.fc(fused)

    @torch.no_grad()
    def init_from_baseline(self, baseline_corn_head):
        assert isinstance(baseline_corn_head, CornHead)
        self.fc.weight[:, :self.global_dim].copy_(baseline_corn_head.fc.weight)
        self.fc.weight[:, self.global_dim:].zero_()
        self.fc.bias.copy_(baseline_corn_head.fc.bias)


class GlobalLocalFusionModel(nn.Module):
    def __init__(self, encoder="mit_b1", weights=None, num_classes=4, roi_output_size=3, local_dim=128):
        super().__init__()
        self.backbone = MultiStageBackbone(encoder, weights)
        self.local_proj = LocalROIProjection(LOCAL_IN_CHANNELS, local_dim)
        self.fusion_head = GlobalLocalFusionHead(GLOBAL_DIM, local_dim, num_classes)
        self.roi_output_size = roi_output_size

    def forward(self, rgb, boxes):
        """rgb: [B,3,224,224]. boxes: list of length B, each a [1,4] tensor
        (x1,y1,x2,y2) in the SAME 224x224 coordinate system as rgb -- NOT
        pre-scaled to feature-map pixels; roi_align's spatial_scale handles
        that internally."""
        feat_stage3, feat_final = self.backbone(rgb)
        f_global = feat_final.mean(dim=(2, 3))  # [B, 512] -- identical op to baseline's own GAP

        spatial_scale = feat_stage3.shape[-1] / rgb.shape[-1]  # dynamically verified, not hardcoded
        roi_feat = roi_align(
            feat_stage3, boxes, output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=spatial_scale, aligned=True,
        )  # [B, 320, k, k]
        f_local = self.local_proj(roi_feat)  # [B, local_dim]

        fused = torch.cat([f_global, f_local], dim=1)
        logits = self.fusion_head(fused)
        return logits, f_global


def load_rgb_baseline(model, checkpoint_path, device):
    """Loads an nn.Sequential(Backbone, CornHead) checkpoint (task_3's own
    save format): "0." keys -> model.backbone.inner, "1." keys -> a
    standalone CornHead used ONLY to initialize the fusion head's global
    columns (see init_from_baseline)."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    backbone_state = {k[len("0."):]: v for k, v in state_dict.items() if k.startswith("0.")}
    head_state = {k[len("1."):]: v for k, v in state_dict.items() if k.startswith("1.")}
    if not backbone_state or not head_state:
        raise RuntimeError(
            f"{checkpoint_path}: expected keys prefixed '0.' (backbone) and '1.' (head), "
            f"got prefixes: {sorted({k.split('.')[0] for k in state_dict})}"
        )
    model.backbone.inner.load_state_dict(backbone_state, strict=True)
    baseline_head = CornHead(GLOBAL_DIM, model.fusion_head.fc.out_features + 1)
    baseline_head.load_state_dict(head_state, strict=True)
    model.fusion_head.init_from_baseline(baseline_head)
    return baseline_head


def build_model(encoder="mit_b1", weights=None, num_classes=4, roi_output_size=3, local_dim=128):
    return GlobalLocalFusionModel(encoder, weights, num_classes, roi_output_size, local_dim)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_params_m(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_last_stage(model):
    """Phase B (optional): unfreeze only the last MiT-B1 stage (the
    sub-modules producing feats[-1] -- smp's MixVisionTransformerEncoder
    names them patch_embed4/block4/norm4, verified via named_children())
    instead of the whole backbone."""
    encoder = model.backbone.inner.encoder
    for name in ("patch_embed4", "block4", "norm4"):
        for p in getattr(encoder, name).parameters():
            p.requires_grad = True
