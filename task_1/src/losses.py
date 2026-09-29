import segmentation_models_pytorch as smp
import torch.nn as nn

LOSSES = ["ce", "dice", "ce_dice", "focal_dice", "tversky"]


def build_loss(name, ignore_index=None):
    ce_ignore = ignore_index if ignore_index is not None else -100
    ce = nn.CrossEntropyLoss(ignore_index=ce_ignore)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
    focal = smp.losses.FocalLoss(mode="multiclass", ignore_index=ignore_index)
    tversky = smp.losses.TverskyLoss(mode="multiclass", ignore_index=ignore_index, alpha=0.3, beta=0.7)

    if name == "ce":
        return ce
    if name == "dice":
        return dice
    if name == "ce_dice":
        return lambda pred, target: 0.5 * ce(pred, target) + 0.5 * dice(pred, target)
    if name == "focal_dice":
        return lambda pred, target: 0.5 * focal(pred, target) + 0.5 * dice(pred, target)
    if name == "tversky":
        return tversky
    raise ValueError(f"Unknown loss: {name}")
