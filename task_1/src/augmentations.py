import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

POLICIES = ["none", "default", "color", "geometric", "heavy"]

# Padded border pixels are labeled "other" (background, class index 0), the
# semantically correct label for the letterboxed region.
PAD_MASK_VALUE = 0


def build_resize(resolution):
    """Source images come from multiple capture sources and span very
    different sizes/aspect ratios (224x224 up to ~1600x1600, portrait and
    landscape). A plain square Resize would squash/stretch wound shape, so
    we scale the longest side down to `resolution` and letterbox-pad the
    rest instead, preserving aspect ratio."""
    return [
        A.LongestMaxSize(max_size=resolution, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=resolution,
            min_width=resolution,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=PAD_MASK_VALUE,
        ),
    ]


def build_transform(policy, resolution, is_train):
    resize = build_resize(resolution)
    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    to_tensor = ToTensorV2()

    if not is_train:
        return A.Compose([*resize, normalize, to_tensor])

    geo_default = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.7),
    ]
    color_ops = [
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
    ]
    geo_heavy = [
        A.ElasticTransform(alpha=30, sigma=5, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    ]
    cutout = [
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.05, 0.1),
            hole_width_range=(0.05, 0.1),
            p=0.3,
        )
    ]

    if policy == "none":
        ops = []
    elif policy == "default":
        ops = geo_default
    elif policy == "color":
        ops = geo_default + color_ops
    elif policy == "geometric":
        ops = geo_default + geo_heavy
    elif policy == "heavy":
        ops = geo_default + color_ops + geo_heavy + cutout
    else:
        raise ValueError(f"Unknown augmentation policy: {policy}")

    return A.Compose([*resize, *ops, normalize, to_tensor])
