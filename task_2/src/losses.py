import torch.nn as nn
from coral_pytorch.losses import corn_loss


class CornLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, target):
        return corn_loss(logits, target, num_classes=self.num_classes)


def build_loss(head, num_classes):
    if head == "softmax":
        return nn.CrossEntropyLoss()
    if head == "corn":
        return CornLoss(num_classes)
    raise ValueError(f"unknown head {head!r}")
