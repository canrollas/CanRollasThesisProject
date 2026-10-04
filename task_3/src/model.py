import segmentation_models_pytorch as smp

# Only the architectures used by configs/selected.yaml's top-10 picks.
ARCH_MAP = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "linknet": smp.Linknet,
}


def build_model(architecture, encoder, weights, num_classes=3):
    cls = ARCH_MAP[architecture]
    return cls(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=num_classes)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
