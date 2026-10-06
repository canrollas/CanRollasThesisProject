import random

import numpy as np


class ImageAugmentation:
    """Same flip/rotation protocol as task_1/task_3's PairedAugmentation, for
    a single image (no mask to keep in sync with, since this is
    classification)."""

    def __init__(self, hflip_p=0.5, vflip_p=0.5, rotate_p=0.7, rotate_angles=(90, 180, 270)):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate_p = rotate_p
        self.rotate_angles = rotate_angles

    def __call__(self, image):
        if random.random() < self.hflip_p:
            image = np.ascontiguousarray(image[:, ::-1])
        if random.random() < self.vflip_p:
            image = np.ascontiguousarray(image[::-1, :])
        if random.random() < self.rotate_p:
            angle = random.choice(self.rotate_angles)
            k = angle // 90
            image = np.ascontiguousarray(np.rot90(image, k))
        return image
