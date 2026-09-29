import copy

import torch

from .metrics import confusion_from_logits, iou_dice_from_confusion


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, num_classes, device, ignore_index=None):
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        conf += confusion_from_logits(logits, targets, num_classes, ignore_index)
    iou, dice = iou_dice_from_confusion(conf)
    return {
        "miou": iou.mean().item(),
        "mdice": dice.mean().item(),
        "iou_per_class": iou.tolist(),
        "dice_per_class": dice.tolist(),
    }


def run_training(model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn,
                  epochs, device, num_classes, ignore_index=None):
    best_miou = -1.0
    best_state = None
    history = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, num_classes, device, ignore_index)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_miou": val_metrics["miou"]})

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, num_classes, device, ignore_index)
    return model, {"best_val_miou": best_miou, "test": test_metrics, "history": history}
