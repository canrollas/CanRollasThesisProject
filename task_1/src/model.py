import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH_MAP = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "fpn": smp.FPN,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(architecture, encoder, weights, num_classes=3):
    cls = ARCH_MAP[architecture]
    return cls(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=num_classes)


class SegformerMLPHead(nn.Module):
    """Hand-written SegFormer all-MLP decode head (Xie et al., NeurIPS 2021):
    each encoder stage is linearly projected to a shared embedding dim,
    upsampled to the highest-resolution input stage, concatenated, and fused
    with a single 1x1 conv before classification.
    """

    def __init__(self, in_channels, embed_dim=256, num_classes=3, dropout=0.1):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(c, embed_dim) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features):
        target_size = features[0].shape[2:]
        projected = []
        for feat, linear in zip(features, self.linears):
            b, c, h, w = feat.shape
            x = feat.flatten(2).transpose(1, 2)
            x = linear(x)
            x = x.transpose(1, 2).reshape(b, -1, h, w)
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)
        x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        x = self.dropout(x)
        return self.classifier(x)


class SegFormer(nn.Module):
    def __init__(self, encoder_name, encoder_weights, num_classes=3, embed_dim=256):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=3, depth=5, weights=encoder_weights
        )
        # SMP encoders emit a list of feature maps of increasing depth; the
        # last 4 correspond to strides 4/8/16/32, matching the 4 inputs the
        # original SegFormer decode head expects.
        channels = self.encoder.out_channels[-4:]
        self.head = SegformerMLPHead(channels, embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)[-4:]
        out = self.head(features)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


def build_segformer(encoder, weights, num_classes=3, embed_dim=256):
    return SegFormer(encoder, weights, num_classes=num_classes, embed_dim=embed_dim)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
