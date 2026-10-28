"""Device-safe CORN probabilities and class/threshold Grad-CAM.

CORN logits are conditional ordinal decisions, not four class logits.  The
historical classifier decodes the median rank from survival probabilities;
this can differ from the most probable class.  CAM targets use that actual
decoded class, and optionally the true class and each conditional logit.

Grad-CAM is coarse positive gradient-weighted activation evidence.  It is
not an attention matrix, a segmentation, or a causal explanation.  Signed
maps and zero-positive-evidence flags are retained for honest inspection.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

DEFAULT_LAYERS = {"last": 5, "middle": 4}


def corn_outputs(logits):
    """Return differentiable CORN outputs for a floating [B,K-1] tensor.

    ``conditional_probabilities[:,k]`` = P(Y>k | Y>k-1), with the first
    condition unconditional.  Their cumulative products are P(Y>k).
    Class probabilities are adjacent survival differences, evaluated in log
    space to avoid cancellation and saturation when used as CAM targets.
    ``decoded`` follows coral_pytorch's strict survival > 0.5 rule exactly.
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] < 1:
        raise ValueError("CORN logits must be a [batch,classes-1] tensor")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("CORN logits must contain finite floating-point values")
    conditional = torch.sigmoid(logits)
    survival = torch.cumprod(conditional, dim=1)
    log_conditional = F.logsigmoid(logits)
    log_not_conditional = F.logsigmoid(-logits)
    log_survival = torch.cumsum(log_conditional, dim=1)
    log_mass = torch.cat((log_not_conditional[:, :1],
                          log_survival[:, :-1] + log_not_conditional[:, 1:],
                          log_survival[:, -1:]), dim=1)
    # Analytically these masses sum to one. Log-softmax removes only floating
    # point drift while preserving finite log targets for extreme finite z.
    log_probabilities = F.log_softmax(log_mass, dim=1)
    probabilities = log_probabilities.exp()
    return {
        "logits": logits,
        "conditional_probabilities": conditional,
        "survival_probabilities": survival,
        "class_probabilities": probabilities,
        "log_class_probabilities": log_probabilities,
        "decoded": (survival > .5).sum(dim=1),
        "probability_argmax": probabilities.argmax(dim=1),
    }


def _target_labels(targets, batch, n_classes, device):
    labels = torch.as_tensor(targets, device=device)
    if labels.shape != (batch,) or labels.dtype == torch.bool:
        raise ValueError("True targets must contain one integer class per image")
    if labels.is_floating_point() and not torch.equal(labels, labels.round()):
        raise ValueError("True targets must be integer class indices")
    labels = labels.long()
    if not ((labels >= 0) & (labels < n_classes)).all():
        raise ValueError("True target outside the CORN class range")
    return labels


def extract_explanations(model, images, targets=None, include_thresholds=True,
                         layer_indices=None):
    """Extract batched raw/normalized Grad-CAM without changing model state.

    The model must be in eval mode and expose ``model[0].encoder`` returning
    SMP-style [B,C,H,W] feature maps. ``layer_indices`` defaults to the MiT-B1
    final map (index 5, 7x7 at 224px) and middle map (index 4, 14x14).

    Returned tensors are detached on CPU. ``cams[target][layer]`` contains
    ``signed``, ``raw`` (ReLU), ``normalized`` (each image divided by its
    positive maximum), and ``zero``. Targets are ``predicted``, ``true``
    when supplied, and ``threshold_0`` ... ``threshold_{K-2}``. Class maps
    explain log P(the specified class), threshold maps explain raw z_k.
    All feature maps retain native coarse resolution; callers upsample them.

    Uses autograd.grad on feature tensors, leaving parameter .grad fields and
    checkpoint weights unchanged. It also works inside an outer no_grad.
    """
    if model.training:
        raise ValueError("Grad-CAM requires model.eval() for independent images")
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or not len(images):
        raise ValueError("Images must be a nonempty [B,C,H,W] tensor")
    layers = dict(DEFAULT_LAYERS if layer_indices is None else layer_indices)
    if not layers or len(set(layers.values())) != len(layers):
        raise ValueError("Choose distinct encoder feature-map indices")
    if any(not isinstance(index, int) or index < 0 for index in layers.values()):
        raise ValueError("Feature-map indices must be nonnegative integers")
    try:
        encoder = model[0].encoder
    except (TypeError, AttributeError, IndexError) as exc:
        raise ValueError("Expected model[0].encoder with SMP feature maps") from exc
    captured = {}

    def capture(_module, _inputs, outputs):
        if not isinstance(outputs, (list, tuple)):
            raise ValueError("Encoder must return an ordered list of feature maps")
        for name, index in layers.items():
            if index >= len(outputs):
                raise ValueError(f"Encoder has no feature index {index}")
            activation = outputs[index]
            if activation.ndim != 4 or activation.shape[1] < 1:
                raise ValueError(f"{name} must be a nonempty [B,C,H,W] feature map")
            captured[name] = activation

    handle = encoder.register_forward_hook(capture)
    try:
        with torch.inference_mode(False), torch.enable_grad():
            inputs = images.detach().clone().requires_grad_(True)
            logits = model(inputs)
            outputs = corn_outputs(logits)
            batch, n_classes = len(inputs), logits.shape[1] + 1
            target_classes = {"predicted": outputs["decoded"].detach()}
            if targets is not None:
                target_classes["true"] = _target_labels(targets, batch, n_classes, logits.device)
            target_scores = {
                name: outputs["log_class_probabilities"].gather(1, labels[:, None]).squeeze(1)
                for name, labels in target_classes.items()
            }
            descriptions = {name: "log P(Y=target_class), using CORN class masses"
                            for name in target_classes}
            if include_thresholds:
                for index in range(n_classes - 1):
                    name = f"threshold_{index}"
                    target_scores[name] = logits[:, index]
                    target_classes[name] = torch.full((batch,), index, device=logits.device, dtype=torch.long)
                    descriptions[name] = (
                        f"raw conditional ordinal logit z_{index}: "
                        f"P(Y>{index} | Y>{index - 1}), zero-based rank"
                    )
            activations = tuple(captured[name] for name in layers)
            cams = {}
            score_items = tuple(target_scores.items())
            for number, (target_name, scores) in enumerate(score_items):
                gradients = torch.autograd.grad(scores.sum(), activations,
                                                retain_graph=number < len(score_items) - 1,
                                                allow_unused=True)
                target_cams = {}
                for (name, activation), gradient in zip(captured.items(), gradients):
                    if gradient is None:
                        signed = activation.new_zeros((batch, *activation.shape[2:]))
                    else:
                        weights = gradient.mean(dim=(-2, -1), keepdim=True)
                        signed = (weights * activation).sum(dim=1)
                    raw = signed.relu()
                    maximum = raw.flatten(1).amax(1)
                    zero = maximum <= 0
                    normalized = raw / maximum.clamp_min(torch.finfo(raw.dtype).tiny)[:, None, None]
                    target_cams[name] = {"signed": signed.detach().cpu(),
                                         "raw": raw.detach().cpu(),
                                         "normalized": normalized.detach().cpu(),
                                         "zero": zero.detach().cpu()}
                cams[target_name] = target_cams
            result = {key: value.detach().cpu() for key, value in outputs.items()}
            result.update({"target_classes": {key: value.detach().cpu() for key, value in target_classes.items()},
                           "cams": cams,
                           "metadata": {"method": "Grad-CAM (positive activation evidence)",
                                        "class_target": "log class probability",
                                        "class_decoder": "CORN median: count(survival > 0.5)",
                                        "layers": layers, "targets": descriptions,
                                        "normalization": "per-image positive maximum; no shared evidence magnitude",
                                        "interpretation": "coarse gradient-weighted evidence; not attention, segmentation or causal proof"}})
            return result
    finally:
        handle.remove()

