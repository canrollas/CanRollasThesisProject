"""Tissue-conditioned spatial RGB tokens with relative inverse-depth bins.

The original MiT-B1/CORN global pathway stays intact. A small spatial branch
reads RGB features under five soft tissue regions and three soft geometry
bins, then answers the three CORN conditional questions with learned queries.
Monocular inverse depth is only a relative appearance prior, not metric wound
depth. Missing geometry reduces exactly to the tissue-only computation.
"""
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F


LAST_STAGE_NAMES = ("patch_embed4", "block4", "norm4")
MODES = ("full", "tissue", "geometry", "rgb")


class SpatialRGBBackbone(nn.Module):
    """State-dict compatible with task_3.src.model.Backbone's SMP path."""

    def __init__(self, encoder="mit_b1", weights=None):
        super().__init__()
        if encoder != "mit_b1":
            raise ValueError("This experiment fixes the encoder to mit_b1")
        # Tests inject a tiny backbone and never import SMP or download weights.
        import segmentation_models_pytorch as smp

        self.encoder = smp.encoders.get_encoder(
            encoder, in_channels=3, depth=5, weights=weights,
        )
        self.out_dim = self.encoder.out_channels[-1]
        self.stage3_dim = self.encoder.out_channels[-2]


class CornHead(nn.Module):
    def __init__(self, in_dim, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes - 1)

    def forward(self, features):
        return self.fc(features)


def relative_depth_geometry(depth, support, depth_valid=None):
    """Return affine-invariant relative inverse depth and per-image validity.

    The reference is the mean inverse depth in a ring outside the predicted
    wound; the scale is the within-image standard deviation over wound+ring.
    Positive scale and arbitrary offset of the input therefore cancel. A
    missing ring, flat map, absent wound, explicit invalid flag, or any
    nonfinite depth pixel disables geometry for that sample.
    """
    if support.ndim != 4 or support.shape[1] != 1:
        raise ValueError("support must have shape [B,1,H,W]")
    batch, _, height, width = support.shape
    zero = torch.zeros_like(support)
    invalid = torch.zeros(batch, device=support.device, dtype=torch.bool)
    if depth is None:
        return zero, invalid
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4 or depth.shape[:2] != (batch, 1):
        raise ValueError("depth must have shape [B,1,H,W] or [B,H,W]")
    depth = depth.to(device=support.device, dtype=support.dtype)
    finite = torch.isfinite(depth).flatten(1).all(dim=1)
    safe = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if safe.shape[-2:] != (height, width):
        safe = F.interpolate(safe, (height, width), mode="bilinear", align_corners=False)
    wound = (support > 0.05).to(support.dtype)
    radius = max(1, round(min(height, width) * 0.04))
    dilated = F.max_pool2d(wound, 2 * radius + 1, stride=1, padding=radius)
    ring = (dilated - wound).clamp_min(0)
    domain = (wound + ring).clamp_max(1)
    ring_count = ring.sum(dim=(1, 2, 3))
    area_count = domain.sum(dim=(1, 2, 3))
    ring_mean = (safe * ring).sum(dim=(1, 2, 3)) / ring_count.clamp_min(1)
    area_mean = (safe * domain).sum(dim=(1, 2, 3)) / area_count.clamp_min(1)
    variance = ((safe - area_mean[:, None, None, None]).square() * domain).sum(
        dim=(1, 2, 3)
    ) / area_count.clamp_min(1)
    scale = variance.sqrt()
    valid = finite & (ring_count > 0) & (wound.flatten(1).sum(1) > 0) & (scale > 1e-6)
    if depth_valid is not None:
        supplied = torch.as_tensor(depth_valid, device=support.device, dtype=torch.bool)
        if supplied.numel() != batch:
            raise ValueError("depth_valid must contain exactly one flag per image")
        valid = valid & supplied.reshape(batch)
    relative = (safe - ring_mean[:, None, None, None]) / scale[:, None, None, None].clamp_min(1e-6)
    relative = torch.where(valid[:, None, None, None], relative.clamp(-4, 4), zero)
    return relative, valid


class TissueGeometryTokenModel(nn.Module):
    def __init__(self, encoder="mit_b1", weights=None, num_classes=4, token_dim=96,
                 mode="full", backbone=None, dropout=0.1, residual_scale=0.2,
                 relation_block=True):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if token_dim % 4:
            raise ValueError("token_dim must be divisible by four attention heads")
        if num_classes != 4:
            raise ValueError("This experiment fixes four staging classes")
        self.mode = mode
        self.residual_scale = float(residual_scale)
        self.rgb_backbone = backbone if backbone is not None else SpatialRGBBackbone(encoder, weights)
        self.rgb_head = CornHead(self.rgb_backbone.out_dim, num_classes)
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(self.rgb_backbone.stage3_dim + self.rgb_backbone.out_dim, token_dim, 1),
            nn.GroupNorm(4, token_dim), nn.GELU(),
        )
        # Metadata: frame abundance, wound share, relative-depth mean/std,
        # confidence, geometry-valid flag, spatial centroid and spatial std.
        self.token_projection = nn.Sequential(
            nn.Linear(2 * token_dim + 10, token_dim), nn.LayerNorm(token_dim),
            nn.GELU(), nn.Linear(token_dim, token_dim),
        )
        self.tissue_embeddings = nn.Parameter(torch.empty(5, token_dim))
        self.bin_embeddings = nn.Parameter(torch.empty(3, token_dim))
        self.null_token = nn.Parameter(torch.empty(1, 1, token_dim))
        self.ordinal_queries = nn.Parameter(torch.empty(1, num_classes - 1, token_dim))
        for param in (self.tissue_embeddings, self.bin_embeddings, self.null_token, self.ordinal_queries):
            nn.init.normal_(param, std=0.02)
        self.relations = nn.TransformerEncoderLayer(
            token_dim, 4, dim_feedforward=token_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        ) if relation_block else None
        self.query_attention = nn.MultiheadAttention(token_dim, 4, dropout=dropout, batch_first=True)
        self.local_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        # Nonzero weights let the spatial branch receive gradients immediately.
        nn.init.normal_(self.local_head[-1].weight, std=0.01)
        nn.init.zeros_(self.local_head[-1].bias)
        self._last_stage_trainable = False
        self.set_last_stage_trainable(False)

    def set_last_stage_trainable(self, trainable=True):
        self._last_stage_trainable = bool(trainable)
        for parameter in self.rgb_backbone.parameters():
            parameter.requires_grad = False
        if trainable:
            for name in LAST_STAGE_NAMES:
                module = getattr(self.rgb_backbone.encoder, name, None)
                if module is None:
                    raise ValueError(f"backbone encoder has no last-stage module {name!r}")
                for parameter in module.parameters():
                    parameter.requires_grad = True
        for parameter in self.rgb_head.parameters():
            parameter.requires_grad = True
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # MiT drop-path/dropout in frozen stages must stay deterministic.
        self.rgb_backbone.eval()
        if mode and self._last_stage_trainable:
            for name in LAST_STAGE_NAMES:
                getattr(self.rgb_backbone.encoder, name).train(True)
        return self

    def load_rgb_baseline(self, checkpoint_path):
        state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise RuntimeError("Expected a nonempty Sequential baseline state_dict")
        if any(not key.startswith(("0.", "1.")) for key in state):
            raise RuntimeError("Expected ONLY '0.' backbone and '1.' head checkpoint keys")
        backbone_state = {key[2:]: value for key, value in state.items() if key.startswith("0.")}
        head_state = {key[2:]: value for key, value in state.items() if key.startswith("1.")}
        if not backbone_state or not head_state:
            raise RuntimeError("Checkpoint must contain both backbone ('0.') and CORN head ('1.')")
        self.rgb_backbone.load_state_dict(backbone_state, strict=True)
        self.rgb_head.load_state_dict(head_state, strict=True)
        return self

    def _tokens(self, features, tissue, relative_depth, geometry_valid):
        batch, channels, height, width = features.shape
        tissue = F.interpolate(tissue, (height, width), mode="bilinear", align_corners=False)
        support = tissue.sum(dim=1, keepdim=True).clamp(0, 1)
        probabilities = tissue / tissue.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / math.log(5)
        confidence = (0.25 + 0.75 * (1 - entropy).clamp(0, 1)) * (support > 0).to(features.dtype)
        geometry = F.interpolate(relative_depth, (height, width), mode="bilinear", align_corners=False)
        centers = features.new_tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1)
        bins = (-((geometry - centers) / 0.75).square()).softmax(dim=1)
        bins = torch.where(geometry_valid[:, None, None, None], bins, torch.full_like(bins, 1 / 3))
        # [B, tissue, depth-bin, H, W] -> 15 uncertainty-weighted regions.
        # Abundance/share below measure confidence-weighted tissue mass; the
        # denominator for wound share remains the full predicted wound area.
        weights = (tissue[:, :, None] * bins[:, None] * confidence[:, None]).reshape(
            batch, 15, height, width,
        )
        mass = weights.sum(dim=(2, 3))
        normalized = weights / mass[:, :, None, None].clamp_min(1e-8)

        def moments(values):
            mean = torch.einsum("bnhw,bchw->bnc", normalized, values)
            second = torch.einsum("bnhw,bchw->bnc", normalized, values.square())
            std = (second - mean.square()).clamp_min(1e-6).sqrt()
            return mean, std

        rgb_mean, rgb_std = moments(features)
        depth_mean, depth_std = moments(geometry)
        depth_mean = depth_mean * geometry_valid[:, None, None]
        depth_std = depth_std * geometry_valid[:, None, None]
        conf_mean, _ = moments(confidence)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=features.device, dtype=features.dtype),
            torch.linspace(-1, 1, width, device=features.device, dtype=features.dtype), indexing="ij",
        )
        coordinates = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        centroid, spatial_std = moments(coordinates)
        abundance = (mass / (height * width)).unsqueeze(-1)
        wound_share = (mass / support.sum(dim=(1, 2, 3))[:, None].clamp_min(1e-8)).unsqueeze(-1)
        geometry_flag = geometry_valid[:, None, None].expand(-1, 15, 1).to(features.dtype)
        metadata = torch.cat((abundance, wound_share, depth_mean, depth_std, conf_mean,
                              geometry_flag, centroid, spatial_std), dim=-1)
        tokens = self.token_projection(torch.cat((rgb_mean, rgb_std, metadata), dim=-1))
        semantic_id = self.tissue_embeddings[:, None].expand(-1, 3, -1).reshape(15, channels)
        bin_id = self.bin_embeddings[None].expand(5, -1, -1).reshape(15, channels)
        tokens = tokens + semantic_id[None] + bin_id[None] * geometry_valid[:, None, None]
        padding = mass <= 1e-6
        # The null token remains visible even when every wound token is absent.
        tokens = torch.cat((tokens, self.null_token.expand(batch, -1, -1)), dim=1)
        padding = torch.cat((padding, torch.zeros(batch, 1, dtype=torch.bool, device=features.device)), dim=1)
        return tokens, padding

    def forward(self, rgb, tissue, depth=None, depth_valid=None):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B,3,H,W]")
        if tissue.ndim != 4 or tissue.shape[0] != rgb.shape[0] or tissue.shape[1] != 5:
            raise ValueError("tissue must have shape [B,5,H,W]")
        if tissue.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("rgb and tissue must share the same spatial coordinates and shape")
        tissue = torch.nan_to_num(tissue.to(device=rgb.device, dtype=rgb.dtype)).clamp(0, 1)
        support = tissue.sum(dim=1, keepdim=True).clamp(0, 1)
        has_wound = support.flatten(1).sum(1) > 1e-6
        maps = self.rgb_backbone.encoder(rgb)
        stage3, stage4 = maps[-2], maps[-1]
        rgb_logits = self.rgb_head(stage4.mean(dim=(2, 3)))
        if self.mode == "rgb":
            return {"logits": rgb_logits, "rgb_logits": rgb_logits,
                    "local_logits": torch.zeros_like(rgb_logits), "has_wound": has_wound,
                    "geometry_valid": torch.zeros_like(has_wound)}
        relative_depth, geometry_valid = relative_depth_geometry(
            depth if self.mode in ("full", "geometry") else None, support,
            depth_valid if self.mode in ("full", "geometry") else None,
        )
        if self.mode == "geometry":
            # Wound+geometry control: retain localization, remove all tissue
            # composition and spatial semantic assignments. This is not an
            # independent depth-only modality because support comes from tissue.
            tissue = support.expand(-1, 5, -1, -1) / 5
        upsampled = F.interpolate(stage4, stage3.shape[-2:], mode="bilinear", align_corners=False)
        spatial = self.spatial_projection(torch.cat((stage3, upsampled), dim=1))
        tokens, padding = self._tokens(spatial, tissue, relative_depth, geometry_valid)
        if self.relations is not None:
            tokens = self.relations(tokens, src_key_padding_mask=padding)
        queries = self.ordinal_queries.expand(rgb.shape[0], -1, -1)
        attended, _ = self.query_attention(queries, tokens, tokens, key_padding_mask=padding, need_weights=False)
        local_logits = self.local_head(attended + queries).squeeze(-1)
        local_logits = local_logits * has_wound[:, None].to(local_logits.dtype)
        logits = rgb_logits + self.residual_scale * local_logits
        return {"logits": logits, "rgb_logits": rgb_logits, "local_logits": local_logits,
                "has_wound": has_wound, "geometry_valid": geometry_valid}


def build_model(**kwargs):
    return TissueGeometryTokenModel(**kwargs)


def load_rgb_baseline(model, checkpoint_path, device=None):
    model.load_rgb_baseline(checkpoint_path)
    if device is not None:
        model.to(device)
    return model


def count_params_m(model):
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6
