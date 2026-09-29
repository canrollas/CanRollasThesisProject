# Datasets

This directory contains the image datasets used in this thesis project for
automated wound assessment. The data is organized into three tasks: wound
region segmentation, wound tissue segmentation, and wound stage
classification. Each subdirectory is self-contained and can be used
independently.

## 1. `wound-region-segmentation/`

Semantic segmentation of wound photographs into three regions: wound,
surrounding skin, and background.

- **`images/`** — 3,476 RGB photographs (224×224 PNG).
- **`masks/`** — 3,476 color-coded segmentation masks, one per image, with
  matching filenames.
- **`color-mappings.json`** — maps each class to the RGB color used in the
  masks:

  | Class   | Color     |
  |---------|-----------|
  | other   | `#000000` |
  | skin    | `#004CFF` |
  | wound   | `#FF0000` |

## 2. `wound-tissue-segmentation/`

Semantic segmentation of wound tissue types within the wound bed.

- **`tissue_images/`** — 612 RGBA photographs (512×512 PNG).
- **`tissue_masks/`** — 612 color-coded segmentation masks, one per image,
  with matching filenames. Filenames follow the pattern
  `<case-uuid>_wound_<n>.png`, where `<n>` indexes distinct wounds imaged
  from the same case.
- **`color-mappings.json`** — maps each class to the RGB color used in the
  masks:

  | Class          | Color     | Notes  |
  |----------------|-----------|--------|
  | slough         | `#FFFF00` |        |
  | necrosis       | `#800080` |        |
  | tendon         | `#00E5FF` |        |
  | bone           | `#00FF00` |        |
  | granulation    | `#FF0000` |        |
  | outside_wound  | `#000000` | ignore |
  | skin_remnant   | `#004CFF` | ignore |

  The `outside_wound` and `skin_remnant` classes mark pixels outside the
  region of interest and should be excluded (e.g., masked out or given zero
  weight) when training tissue classifiers.

## 3. `wound-stage-classification/`

Image classification of wounds by severity stage, organized as one
directory per class (1,091 RGB images, 224×224 JPEG).

| Stage     | Directory  | Images |
|-----------|------------|--------|
| Stage 1   | `stage_1/` | 230    |
| Stage 2   | `stage_2/` | 313    |
| Stage 3   | `stage_3/` | 275    |
| Stage 4   | `stage_4/` | 273    |

## Notes

- All images are de-identified clinical wound photographs and are provided
  for research use within this project only.
- File and directory names encode the sample identifier (and, where
  applicable, case UUID and wound index) needed to trace an image back to
  its corresponding mask or label.
