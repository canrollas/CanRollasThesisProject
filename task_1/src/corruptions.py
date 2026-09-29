import albumentations as A

# Test-time-only corruptions used by run_robustness.py to probe a fixed,
# already-trained checkpoint against realistic capture-condition shifts
# (noisy sensors, poor focus, bad clinic lighting, compression, occlusion
# from gauze/tape).
CORRUPTIONS = {
    "clean": A.NoOp(),
    "gaussian_noise": A.GaussNoise(std_range=(0.08, 0.2), p=1.0),
    "gaussian_blur": A.GaussianBlur(blur_limit=(5, 9), p=1.0),
    "brightness_contrast": A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
    "jpeg_compression": A.ImageCompression(quality_range=(20, 40), p=1.0),
    "occlusion": A.CoarseDropout(
        num_holes_range=(2, 6),
        hole_height_range=(0.08, 0.15),
        hole_width_range=(0.08, 0.15),
        p=1.0,
    ),
}
