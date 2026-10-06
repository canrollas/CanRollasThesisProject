import random

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image


class ImageAugmentation:
    """Flip/rotation protocol (shared with task_1/task_3's PairedAugmentation)
    plus continuous affine jitter and mild photometric jitter, modelled after
    pu-stagenet's recipe for this same dataset: colour *is* the pathology here
    (red granulation vs yellow slough vs black necrosis), so hue jitter stays
    tiny and low-probability instead of the usual ColorJitter(hue=0.1), which
    would destroy the signal it's supposed to regularise.
    """

    def __init__(self, hflip_p=0.5, vflip_p=0.5, rotate_p=0.7, rotate_angles=(90, 180, 270),
                 affine_p=0.8, max_rotate_deg=25, scale_range=(0.85, 1.18), translate_frac=0.07,
                 brightness_range=(0.82, 1.18), contrast_range=(0.82, 1.18),
                 hue_p=0.3, hue_shift_deg=8):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate_p = rotate_p
        self.rotate_angles = rotate_angles
        self.affine_p = affine_p
        self.max_rotate_deg = max_rotate_deg
        self.scale_range = scale_range
        self.translate_frac = translate_frac
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.hue_p = hue_p
        self.hue_shift_deg = hue_shift_deg

    def __call__(self, image):
        if random.random() < self.hflip_p:
            image = np.ascontiguousarray(image[:, ::-1])
        if random.random() < self.vflip_p:
            image = np.ascontiguousarray(image[::-1, :])
        if random.random() < self.rotate_p:
            angle = random.choice(self.rotate_angles)
            k = angle // 90
            image = np.ascontiguousarray(np.rot90(image, k))

        if random.random() < self.affine_p:
            h, w = image.shape[:2]
            angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
            scale = random.uniform(*self.scale_range)
            max_dx, max_dy = self.translate_frac * w, self.translate_frac * h
            translate = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
            # fill with this image's mean colour rather than black, so the
            # corners exposed by rotation/scale don't look like a hard edge
            fill = tuple(int(image[..., c].mean()) for c in range(image.shape[2]))
            pil = Image.fromarray(image)
            pil = TF.affine(pil, angle=angle, translate=translate, scale=scale, shear=0.0,
                             interpolation=TF.InterpolationMode.BILINEAR, fill=fill)
            image = np.array(pil)

        x = image.astype(np.float32) / 255.0
        x = x * random.uniform(*self.brightness_range)
        mean = x.mean()
        x = (x - mean) * random.uniform(*self.contrast_range) + mean
        x = np.clip(x, 0.0, 1.0)
        image = (x * 255.0).astype(np.uint8)

        if random.random() < self.hue_p:
            hue_factor = random.uniform(-self.hue_shift_deg, self.hue_shift_deg) / 360.0
            pil = TF.adjust_hue(Image.fromarray(image), hue_factor)
            image = np.array(pil)

        return image
