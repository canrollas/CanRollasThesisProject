import segmentation_models_pytorch as smp

DECODERS = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "fpn": smp.FPN,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "pan": smp.PAN,
    "pspnet": smp.PSPNet,
    "deeplabv3": smp.DeepLabV3,
    "segformer": smp.Segformer,
}


def build_model(decoder, encoder, weights, num_classes):
    if decoder not in DECODERS:
        raise ValueError(f"Unknown decoder: {decoder}. Valid options: {list(DECODERS)}")
    model_cls = DECODERS[decoder]
    encoder_weights = None if weights in (None, "none") else weights
    return model_cls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=None,
    )
