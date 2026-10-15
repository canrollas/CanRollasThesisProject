import torch
import torch.nn as nn
import torch.nn.functional as F

VOID_LABEL = 255


class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1.0, ignore_index=VOID_LABEL):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        valid = (target != self.ignore_index).unsqueeze(1).float()
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0

        probs = F.softmax(logits, dim=1) * valid
        target_onehot = F.one_hot(target_clamped, self.num_classes).permute(0, 3, 1, 2).float() * valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """0.5 * CrossEntropy + 0.5 * Dice, per the paper's training protocol.
    Both terms exclude void-labelled pixels (outside_wound/skin_remnant/
    tendon/bone, see src/dataset.py).
    """

    def __init__(self, num_classes=3, ce_weight=0.5, dice_weight=0.5, ignore_index=VOID_LABEL):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)
