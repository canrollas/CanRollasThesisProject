#!/usr/bin/env python3
"""Exp12 model: MiT-B2 encoder initialized from Task 2's own segmentation-
trained checkpoint (MAnet+MiT-B2, tissue-type segmentation: granulation /
slough / necrosis / tendon / bone), decoder discarded, CORN ordinal head
added on top -- then FULLY fine-tuned on the PIID staging dataset, same as
every other config in task_3's own ablation grid. No freezing: this is a
weight-initialization experiment, not a residual/frozen-backbone one like
exp10.

Architecturally this IS task_3's own build_model("mit_b2", ..., "corn")
(src/model.py): the only difference is where the encoder's weights come
from before fine-tuning starts. Key compatibility verified directly --
smp.MAnet's `encoder.*` state-dict keys are byte-identical in name and
shape to task_3's Backbone.encoder (both wrap
smp.encoders.get_encoder("mit_b2", ...) the same way): 332/332 keys
match, "All keys matched successfully" on load_state_dict. See README's
"Mimari doğrulama" section for the exact check transcript.
"""
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "task_3"))

from src.model import Backbone, CornHead  # noqa: E402


def load_task2_encoder(backbone, task2_checkpoint_path, device):
    """Loads Task 2's MAnet_mit_b2 checkpoint's `encoder.*` submodule into
    `backbone.encoder` in place. Returns a small verification dict so the
    caller can assert the transfer actually did something before training
    (not a silent no-op / partial load)."""
    ckpt = torch.load(task2_checkpoint_path, map_location=device)
    enc_keys = [k for k in ckpt if k.startswith("encoder.")]
    if not enc_keys:
        raise ValueError(f"no 'encoder.*' keys found in {task2_checkpoint_path} -- wrong checkpoint format?")
    enc_state = {k[len("encoder."):]: v for k, v in ckpt.items() if k.startswith("encoder.")}

    probe_key = next(iter(backbone.encoder.state_dict()))
    before = backbone.encoder.state_dict()[probe_key].clone()
    backbone.encoder.load_state_dict(enc_state)  # SMP's MixVisionTransformerEncoder overrides
    # load_state_dict with no `strict` kwarg; it always raises on a mismatched key set.
    after = backbone.encoder.state_dict()[probe_key]

    return {
        "checkpoint_encoder_keys": len(enc_keys),
        "backbone_encoder_keys": len(backbone.encoder.state_dict()),
        "weights_changed_from_init": not torch.equal(before, after),
    }


def build_model(task2_checkpoint_path, device, num_classes=4, encoder_name="mit_b2"):
    """weights=None: the architecture is built with random init, then
    immediately and fully overwritten by load_task2_encoder below -- no
    point downloading ImageNet weights just to discard them."""
    backbone = Backbone(encoder_name, weights=None)
    verify = load_task2_encoder(backbone, task2_checkpoint_path, device)
    head = CornHead(backbone.out_dim, num_classes)
    model = torch.nn.Sequential(backbone, head)
    return model, verify
