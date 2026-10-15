import numpy as np
import torch
from coral_pytorch.dataset import corn_label_from_logits
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


@torch.no_grad()
def predict_labels(logits, head):
    if head == "softmax":
        return logits.argmax(dim=1)
    if head == "corn":
        return corn_label_from_logits(logits)
    raise ValueError(f"unknown head {head!r}")


def compute_metrics(y_true, y_pred, num_classes=4):
    """accuracy/macro_f1 are the standard nominal-classification metrics;
    mae (mean absolute stage-index error) and qwk (quadratic weighted kappa)
    are the ordinal-aware ones CORN is expected to actually move, since they
    penalize a stage_1-vs-stage_4 mistake more than a stage_1-vs-stage_2 one.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }
