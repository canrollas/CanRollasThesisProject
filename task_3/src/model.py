import segmentation_models_pytorch as smp
import timm
import torch.nn as nn

# Covered by segmentation_models_pytorch's encoder registry (same names used
# in task_1/configs/grid.yaml), reused here purely as feature extractors
# (global-average-pooled last feature map -> classification head, no decoder).
SMP_ENCODERS = {
    "resnet18", "resnet34", "resnet50",
    "efficientnet-b0", "efficientnet-b3", "efficientnet-b4",
    "mobilenet_v2", "densenet121", "resnext50_32x4d",
    "mit_b1", "mit_b2", "mit_b3",
}

# Not in segmentation_models_pytorch's registry, built directly via timm
# instead (timm's num_classes=0 strips the classification head and returns
# globally-pooled features, same shape of output as the SMP path above).
TIMM_ENCODERS = {
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
}


class Backbone(nn.Module):
    def __init__(self, encoder_name, weights):
        super().__init__()
        if encoder_name in SMP_ENCODERS:
            self.kind = "smp"
            self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, depth=5, weights=weights)
            self.out_dim = self.encoder.out_channels[-1]
        elif encoder_name in TIMM_ENCODERS:
            self.kind = "timm"
            self.encoder = timm.create_model(
                TIMM_ENCODERS[encoder_name], pretrained=(weights is not None),
                num_classes=0, global_pool="avg",
            )
            self.out_dim = self.encoder.num_features
        else:
            raise ValueError(f"unknown encoder {encoder_name!r}")

    def forward(self, x):
        if self.kind == "smp":
            last_feature_map = self.encoder(x)[-1]
            return last_feature_map.mean(dim=(2, 3))
        return self.encoder(x)


class SoftmaxHead(nn.Module):
    """Plain nominal classification head: one logit per class, no ordinal
    structure (paired with nn.CrossEntropyLoss in src/losses.py)."""

    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class CornHead(nn.Module):
    """CORN ordinal head (Shi et al., 2022, "Deep Neural Networks for
    Rank-Consistent Ordinal Regression Based on Conditional Probabilities"):
    outputs num_classes-1 conditional binary logits instead of num_classes
    nominal logits. Paired with coral_pytorch's corn_loss /
    corn_label_from_logits in src/losses.py and src/metrics.py.
    """

    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes - 1)

    def forward(self, x):
        return self.fc(x)


HEAD_MAP = {"softmax": SoftmaxHead, "corn": CornHead}


def build_model(encoder, weights, head, num_classes=4):
    backbone = Backbone(encoder, weights)
    head_module = HEAD_MAP[head](backbone.out_dim, num_classes)
    return nn.Sequential(backbone, head_module)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
