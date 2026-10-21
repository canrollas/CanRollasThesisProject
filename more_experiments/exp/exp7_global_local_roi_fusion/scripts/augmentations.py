"""Experiment 7: image+ROI-box paired augmentation.

task_3/src/augmentations.py's ImageAugmentation applies hflip/vflip/rotate
to the RGB image ALONE (fine for plain classification, no mask to keep in
sync). Here the ROI box must stay geometrically aligned with the image
content after the SAME random flip/rotation -- otherwise the box would
point at the wrong region of the augmented image. This mirrors
task_1/src/augmentations.py's PairedAugmentation (image+mask) pattern,
with the box's corner coordinates transformed by the same formulas instead
of a second raster mask.

Coordinate formulas for a square canvas of size S (box corners in
continuous [0,S] space, same np.rot90/slice-flip semantics as the image
ops below, derived and verified empirically against np.rot90's actual
corner mapping -- not assumed):
  hflip (image[:, ::-1]):           x' = S - x      (y unchanged)
  vflip (image[::-1, :]):           y' = S - y      (x unchanged)
  rot90 k=1 (90 deg):   (x, y) -> (x'=y,     y'=S-x)
  rot90 k=2 (180 deg):  (x, y) -> (x'=S-x,   y'=S-y)
  rot90 k=3 (270 deg):  (x, y) -> (x'=S-y,   y'=x)
Applied to both box corners, then re-sorted into (x1,y1,x2,y2) axis-aligned
form (90-degree-multiple transforms always preserve axis alignment, only
possibly swapping which corner is the min/max).
"""
import random

import numpy as np


def _transform_box(box, op, size):
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y2)]
    if op == "hflip":
        corners = [(size - x, y) for x, y in corners]
    elif op == "vflip":
        corners = [(x, size - y) for x, y in corners]
    elif op == "rot1":
        corners = [(y, size - x) for x, y in corners]
    elif op == "rot2":
        corners = [(size - x, size - y) for x, y in corners]
    elif op == "rot3":
        corners = [(size - y, x) for x, y in corners]
    else:
        raise ValueError(op)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


class PairedImageBoxAugmentation:
    """Applies identical spatial augmentation to an image/ROI-box pair.
    Same protocol/hyperparameters as task_3's ImageAugmentation (independent
    h-flip, v-flip at p=0.5 each, then a {90,180,270} rotation at p=0.7) --
    just also keeping the box in sync, like task_1's PairedAugmentation
    keeps an image/mask pair in sync."""

    def __init__(self, hflip_p=0.5, vflip_p=0.5, rotate_p=0.7, rotate_angles=(90, 180, 270), size=224):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate_p = rotate_p
        self.rotate_angles = rotate_angles
        self.size = size

    def __call__(self, image, box):
        if random.random() < self.hflip_p:
            image = np.ascontiguousarray(image[:, ::-1])
            box = _transform_box(box, "hflip", self.size)
        if random.random() < self.vflip_p:
            image = np.ascontiguousarray(image[::-1, :])
            box = _transform_box(box, "vflip", self.size)
        if random.random() < self.rotate_p:
            angle = random.choice(self.rotate_angles)
            k = angle // 90
            image = np.ascontiguousarray(np.rot90(image, k))
            box = _transform_box(box, f"rot{k}", self.size)
        return image, box
