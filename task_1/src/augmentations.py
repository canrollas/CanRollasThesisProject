import random

import numpy as np


class PairedAugmentation:
    """Applies identical spatial augmentation to an image/mask pair, matching
    the paper's protocol: independent h-flip and v-flip (p=0.5 each), then a
    random rotation from {90, 180, 270} degrees (p=0.7). No color/photometric
    augmentation.
    """

    def __init__(self, hflip_p=0.5, vflip_p=0.5, rotate_p=0.7, rotate_angles=(90, 180, 270)):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate_p = rotate_p
        self.rotate_angles = rotate_angles

    def __call__(self, image, mask):
        if random.random() < self.hflip_p:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < self.vflip_p:
            image = np.ascontiguousarray(image[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])
        if random.random() < self.rotate_p:
            angle = random.choice(self.rotate_angles)
            k = angle // 90
            image = np.ascontiguousarray(np.rot90(image, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
        return image, mask
