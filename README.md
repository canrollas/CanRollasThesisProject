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
| 2 | Wound stage classification (Stage 1–4) | Image classification | 🚧 Scaffolded — [`task_2/`](task_2) (pending run) |
| 3 | Wound tissue segmentation (slough, necrosis, granulation, ...) | Semantic segmentation | ✅ Implemented — [`task_3/`](task_3) |

## Repository structure

```
.
├── datasets/           # Clinical image datasets (not tracked in git — see below)
│   └── README.md       # Full dataset documentation (classes, sizes, color maps)
├── task_1/              # Wound region segmentation pipeline
│   ├── configs/         # Ablation grid (architecture x encoder x weights) + training hyperparameters
│   ├── src/             # Dataset, models, losses, training engine, metrics
│   ├── check_compatibility.py  # Resolves the architecture/encoder grid into valid configs
│   ├── run_ablation.py         # Trains + evaluates every (config, fold) combination
│   └── run_efficiency.py       # Measures params and inference latency per config
├── task_2/              # Wound stage classification pipeline (encoder x head ablation, 3-fold)
│   ├── configs/         # Encoder x head grid (softmax vs CORN ordinal) + training hyperparameters
│   ├── src/             # Dataset, models (SMP/timm backbones + softmax/CORN heads), training engine
│   ├── build_grid.py           # Resolves configs/grid.yaml into configs/resolved_grid.json
│   ├── run_ablation.py         # Trains + evaluates every (config, fold) combination
│   └── run_efficiency.py       # Measures params and inference latency per config
└── task_3/              # Wound tissue segmentation pipeline (top-10 configs, 3-fold)
    ├── configs/         # Hand-picked top-10 (architecture, encoder, weights) + training hyperparameters
    ├── src/             # Dataset, models, void-aware losses/metrics, training engine
    ├── build_selected_configs.py  # Resolves configs/selected.yaml into selected_configs.json
    ├── run_ablation.py            # Trains + evaluates every (config, fold) combination
    └── run_efficiency.py          # Measures params and inference latency per config
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

After every completed `(config, fold)`, a 3-panel PNG (image / ground truth
/ prediction overlay) is saved to `results/samples/`, so segmentation
quality can be checked visually without re-running inference.

[`task_1/task1_colab_ablation.ipynb`](task_1/task1_colab_ablation.ipynb) is a ready-to-run
Colab notebook: mounts Drive, clones/pulls this repo, unzips the dataset,
redirects checkpoints/results/samples to Drive for persistence across
sessions, and previews the saved sample panels inline.

## Task 2 — Wound Stage Classification

Ablation over **backbone encoder × classification head**: every encoder is
trained once with a plain nominal softmax head and once with a CORN ordinal
head (Shi et al., 2022), under identical conditions, to test whether treating
Stage 1–4 as an ordered label (rather than four unrelated classes) improves
results — expected to show up mainly in the ordinal-aware metrics (MAE, QWK)
rather than raw accuracy.

- **Encoders:** ResNet-{18,34,50}, EfficientNet-{B0,B3,B4}, MobileNetV2,
  DenseNet121, ResNeXt50, MiT-{B1,B2,B3} (same family as Task 1, for
  cross-task comparability), plus ConvNeXtV2-Tiny — 13 encoders × 2 heads =
  26 configs
- **Heads:** softmax (baseline, `nn.CrossEntropyLoss`) vs. CORN ordinal
  regression (`coral_pytorch.losses.corn_loss`) — same backbone, same
  training protocol, only the final layer and loss differ
- **Evaluation:** 3-fold cross-validation, **stratified per stage** (each
  fold keeps roughly the dataset's 230/313/275/273 stage balance); accuracy,
  macro-F1, MAE (mean absolute stage-index error), and QWK (quadratic
  weighted kappa) per config
- **Augmentation:** horizontal/vertical flip, 90/180/270° rotation (no
  photometric augmentation — wound colour is diagnostically meaningful, same
  rationale as Task 3)

### Reproducing

```bash
cd task_2
python -m venv .venv && source .venv/bin/activate   # or reuse task_1/.venv
pip install -r requirements.txt

python build_grid.py       # resolves configs/grid.yaml -> configs/resolved_grid.json
python run_ablation.py     # trains + evaluates every (config, fold) -> results/ablation_results.csv
python run_efficiency.py   # params + latency per config -> results/efficiency.csv
```

Same interrupt/resume behavior as `task_1/run_ablation.py`:

```bash
python run_ablation.py --configs resnet34__softmax,resnet34__corn
python run_ablation.py --smoke     # 1 config, 1 fold, 2 epochs — sanity check
```

[`task_2/task2_colab_ablation.ipynb`](task_2/task2_colab_ablation.ipynb) is a
ready-to-run Colab notebook: mounts Drive, clones/pulls this repo, unzips the
dataset, redirects checkpoints/results to Drive for persistence across
sessions, and previews the ablation summary leaderboard inline.

> Not yet run end-to-end — the pipeline is scaffolded (dataset, models,
> losses, metrics, ablation runner) but no results exist yet.

## Task 3 — Wound Tissue Segmentation

Re-runs the 10 best-performing configurations from the paper's 37-config
intra-wound tissue segmentation benchmark (`Deep-Learning-Konf.pdf`, Table 5)
on this project's `wound-tissue-segmentation` dataset, under the same 3-fold
protocol, to verify they reproduce on this data before committing further
compute to the full 37-config grid.

- **Selected configs (ranked by the paper's reported mIoU):** MA-Net&MiT-B2,
  U-Net&MiT-B1, U-Net&MiT-B2, U-Net&ResNet-50(SSL), MA-Net&ResNet-34,
  DeepLabV3&ResNet-50(SWSL), U-Net&ResNet-34, DeepLabV3+&MiT-B3,
  U-Net++&ResNet-50, LinkNet&ResNet-34 — see
  [`task_3/configs/selected.yaml`](task_3/configs/selected.yaml)
- **Classes:** granulation, slough/fibrin, necrosis. `outside_wound`,
  `skin_remnant`, `tendon`, and `bone` pixels are treated as void (excluded
  from loss and metrics — this dataset has a `bone` class the paper's
  version didn't annotate)
- **Evaluation:** 3-fold cross-validation (paper protocol: shuffle once with
  a fixed seed, split into 3 consecutive chunks), mIoU / mDice per class
- **Loss:** combined cross-entropy + Dice, both void-aware (`ignore_index`)
- **Augmentation:** horizontal/vertical flip, 90/180/270° rotation

### Reproducing

```bash
cd task_3
python -m venv .venv && source .venv/bin/activate   # or reuse task_1/.venv
pip install -r requirements.txt

python build_selected_configs.py   # resolves configs/selected.yaml -> configs/selected_configs.json
python run_ablation.py             # trains + evaluates every (config, fold) -> results/ablation_results.csv
python run_efficiency.py           # params + latency per config -> results/efficiency.csv
```

Same interrupt/resume behavior as `task_1/run_ablation.py`:

```bash
python run_ablation.py --configs unet__mit_b2__imagenet,manet__mit_b2__imagenet
python run_ablation.py --smoke     # 1 config, 1 fold, 2 epochs — sanity check
```

[`task_3/task3_colab_ablation.ipynb`](task_3/task3_colab_ablation.ipynb) is a ready-to-run
Colab notebook: mounts Drive, clones/pulls this repo, unzips the dataset,
redirects checkpoints/results/samples to Drive for persistence across
sessions, and previews the saved sample panels inline.

## Roadmap

- [x] Task 1: wound region segmentation — architecture/encoder ablation, 3-fold CV
- [ ] Task 2: wound stage classification — encoder x head (softmax/CORN) ablation scaffolded, pending run
- [x] Task 3: wound tissue segmentation — top-10 config re-run, 3-fold CV
- [ ] Cross-task pipeline (region → tissue → stage)

## Author

Can Rollas — thesis project.
