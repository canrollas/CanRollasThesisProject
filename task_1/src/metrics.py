import torch


@torch.no_grad()
def confusion_from_logits(logits, target, num_classes, ignore_index=None):
    pred = logits.argmax(dim=1)
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    idx = target * num_classes + pred
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    return binc.reshape(num_classes, num_classes).cpu()


def iou_dice_from_confusion(conf):
    conf = conf.float()
    tp = conf.diag()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    iou = tp / (tp + fp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    return iou, dice
