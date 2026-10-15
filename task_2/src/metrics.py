import torch

VOID_LABEL = 255


@torch.no_grad()
def compute_confusion(logits, target, num_classes=3, ignore_index=VOID_LABEL):
    preds = torch.argmax(logits, dim=1)
    valid = target != ignore_index
    t = target[valid]
    p = preds[valid]
    idx = t * num_classes + p
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    return binc.reshape(num_classes, num_classes)


def iou_dice_from_confusion(conf):
    """conf: numpy array, rows = ground truth class, cols = predicted class."""
    conf = conf.astype("float64")
    tp = conf.diagonal()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    iou = tp / (tp + fp + fn).clip(min=1e-9)
    dice = 2 * tp / (2 * tp + fp + fn).clip(min=1e-9)
    return iou, dice
