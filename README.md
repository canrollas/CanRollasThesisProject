# Deep Learning Based Wound Region Segmentation and Wound Stage Classification

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![License](https://img.shields.io/badge/data-research--only-lightgrey)

A thesis project on automated wound assessment from clinical photographs,
covering three deep-learning tasks: **wound region segmentation**, **wound
tissue segmentation**, and **wound stage classification**.

## Overview

Chronic wound assessment is normally done by hand — a clinician visually
estimates wound boundaries, tissue composition, and healing stage from a
photograph. This project builds and benchmarks deep-learning models that
automate each of those steps, so they can eventually support (not replace)
clinical judgment.

The work is split into three tasks, each backed by its own dataset:

| # | Task | Type | Status |
|---|------|------|--------|
| 1 | Wound region segmentation (wound / skin / background) | Semantic segmentation | ✅ Implemented — [`task_1/`](task_1) |
| 2 | Wound tissue segmentation (slough, necrosis, granulation, ...) | Semantic segmentation | 🔜 Planned |
| 3 | Wound stage classification (Stage 1–4) | Image classification | 🔜 Planned |

## Repository structure

```
.
├── datasets/           # Clinical image datasets (not tracked in git — see below)
│   └── README.md       # Full dataset documentation (classes, sizes, color maps)
└── task_1/              # Wound region segmentation pipeline
    ├── configs/         # Ablation grid (architecture x encoder x weights) + training hyperparameters
    ├── src/             # Dataset, models, losses, training engine, metrics
    ├── check_compatibility.py  # Resolves the architecture/encoder grid into valid configs
    ├── run_ablation.py         # Trains + evaluates every (config, fold) combination
    └── run_efficiency.py       # Measures params and inference latency per config
```

## Datasets

Three datasets of de-identified clinical wound photographs, one per task —
see [`datasets/README.md`](datasets/README.md) for full details (class
definitions, color mappings, image counts).

| Dataset | Images | Classes |
|---|---|---|
| `wound-region-segmentation/` | 3,476 | background, skin, wound |
| `wound-tissue-segmentation/` | 612 | slough, necrosis, tendon, bone, granulation (+2 ignored) |
| `wound-stage-classification/` | 1,091 | Stage 1–4 |

> The images themselves are **not committed to git** (`.gitignore` excludes
> `datasets/*`) — they're clinical data distributed separately (e.g. via
> Google Drive for Colab). Only `datasets/README.md` is version-controlled.

## Task 1 — Wound Region Segmentation

Replicates a Stage-1 architecture ablation: every combination of **decoder
architecture** × **encoder backbone** × **pretrained weights** is trained
and evaluated under identical conditions, so results are comparable
head-to-head.

- **Decoders:** U-Net, U-Net++, MAnet, FPN, DeepLabV3+, and a hand-written
  SegFormer (all-MLP decode head, MiT encoders only)
- **Encoders:** ResNet-{18,34,50}, EfficientNet-{B0,B3,B4}, MobileNetV2,
  DenseNet121, ResNeXt50, MiT-{B1,B2,B3} — 69 valid (architecture, encoder,
  weights) combinations after compatibility resolution
- **Evaluation:** 3-fold cross-validation, mIoU / mDice per class, param
  count, and inference latency
- **Loss:** combined cross-entropy + Dice
- **Augmentation:** horizontal/vertical flip, 90/180/270° rotation

### Reproducing

```bash
cd task_1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python check_compatibility.py   # resolves configs/grid.yaml -> configs/resolved_grid.json
python run_ablation.py          # trains + evaluates every (config, fold) -> results/ablation_results.csv
python run_efficiency.py        # params + latency per config -> results/efficiency.csv
```

`run_ablation.py` is safe to interrupt and resume at any point — completed
`(config, fold)` rows are skipped on the results CSV, and each fold now
checkpoints every epoch, so a killed run picks back up within a single
epoch instead of restarting the fold. Useful for training on Colab across
multiple, possibly disconnecting, sessions:

```bash
python run_ablation.py --study unet          # run just one architecture family
python run_ablation.py --study segformer_only
python run_ablation.py --smoke               # 1 config, 1 fold, 2 epochs — sanity check
```

## Roadmap

- [x] Task 1: wound region segmentation — architecture/encoder ablation, 3-fold CV
- [ ] Task 2: wound tissue segmentation
- [ ] Task 3: wound stage classification
- [ ] Cross-task pipeline (region → tissue → stage)

## Author

Can Rollas — thesis project.
