import segmentation_models_pytorch as smp

# The architectures used by configs/grid.yaml's 37-config benchmark matrix.
ARCH_MAP = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
}


def build_model(architecture, encoder, weights, num_classes=3):
    cls = ARCH_MAP[architecture]
    return cls(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=num_classes)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
