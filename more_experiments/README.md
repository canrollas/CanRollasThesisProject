# more_experiments — Chapter 8: Wound Staging Improvement

Additional experiments for Chapter 8: can staging (Task 3) be improved by
reusing wound localisation (Task 1) information, without leaking PIID
staging data into the model that produces it? Tissue-composition (Task 2)
experiments come next, once exp-1/exp-2 are evaluated.

## Folder structure

```
more_experiments/
├── manifests/                 verified PIID overlap report + leakage-free Task 1 filelist
├── scripts/
│   └── verify_piid_overlap.py     reproduces manifests/ from raw dataset files, no cached assumptions
├── task1_retrain/              Step 1: leakage-free Task 1 (MAnet+MiT-B2) -- DONE, all 3 folds trained
│   ├── configs/base.yaml
│   ├── leakagefree_splits.py
│   ├── run_leakagefree.py
│   ├── colab_leakagefree_retrain.ipynb
│   └── checkpoints/            manet__mit_b2__imagenet_leakagefree_fold{0,1,2}.pt
├── piid_crops/                 shared Step 2: ensemble-mask generation (feeds every exp/ below)
│   ├── generate_masks_ensemble.py   -> datasets/wound-staging-masks/ (3-fold soft-vote mask)
│   ├── generate_masks.py            single-fold version, kept for fold-vs-fold comparison
│   └── compare_folds.py             per-image mask IoU + detection-rate agreement between folds
└── exp/                         one self-contained folder per staging experiment
    ├── exp1_wound_crop/         skin+wound kept, background dropped
    │   ├── configs/base.yaml
    │   ├── scripts/
    │   │   ├── generate_dataset.py    -> datasets/wound-staging-masks-exp-1/
    │   │   └── train.py               MiT-B1 + CORN (task_3's confirmed best config)
    │   ├── checkpoints/
    │   └── results/
    └── exp2_wound_dilated/      wound only kept, adaptively dilated when small/undetected
        ├── scripts/
        │   └── generate_dataset.py    -> datasets/wound-staging-masks-exp-2/
        ├── checkpoints/         (empty -- train.py not written yet)
        └── results/
```

`checkpoints/`, `results/`, and `task1_retrain/checkpoints/*.pt` are
gitignored (Colab/local-generated, same convention as `task_1/`–`task_3/`).
Original thesis experiments and checkpoints under `task_1/`, `task_2/`,
`task_3/` are untouched by anything here. Future tissue-composition
experiments get their own `exp/expN_.../` folder alongside exp1/exp2 when
we get to them -- same shape, no restructuring needed.

## 1. PIID overlap — verified

Run with `task_1/.venv/bin/python3 more_experiments/scripts/verify_piid_overlap.py`.
Full output: `manifests/piid_overlap_report.json`.

| | |
|---|---|
| Task 1 total images | 3,476 |
| — `fusc` (FUSeg) | 1,210 |
| — `wsnet` (WSNet) | 1,176 |
| — `medetec` | 374 |
| — `ft_tendonbone` (clinical photos) | 488 |
| — `ft_evre1_stage_1` (**PIID Stage 1**) | **228** |
| Leakage-free Task 1 count | **3,248** (= 3,476 − 228, matches the thesis) |

228 confirmed correct (not 221): Task 1's own `src/splits.py` already tags
this subset by filename prefix, and all 228 match, by trailing numeric id,
228 of the 230 images in `datasets/wound-stage-classification/stage_1/` —
spot-checked by opening image pairs, identical photos. The other 2 Stage-1
PIID images were never in Task 1's training set to begin with. (Naive
numeric-id matching against stage_2/3/4 turns up unrelated false-positive
collisions from `fusc`/`wsnet`/`ft_tendonbone` reusing small ids — logged
in the report, not real overlaps.)

Task 2 (tissue segmentation) has **no PIID overlap** at all — its images
share source photos with Task 1's clinical subset, not with PIID staging —
so it needs no leakage-free retrain when we get to it.

## 2. Step 1 — leakage-free Task 1 retrain (DONE)

MAnet + MiT-B2, trained on the 3,248 leakage-free images, 3-fold CV,
reusing `task_1/src` verbatim. All 3 fold checkpoints trained on Colab and
copied into `task1_retrain/checkpoints/`.

Fold-vs-fold comparison on the full PIID staging set (`piid_crops/compare_folds.py`):
stage_2/3/4 are identical across folds (open wounds, easy). stage_1 (intact
or near-intact skin) is where folds disagree — detection rate rises
monotonically fold0 (73.0%) → fold1 (80.0%) → fold2 (89.6%), 58/230 stage_1
images where the three folds don't all agree. Decided to **ensemble all
three** (average softmax probabilities, then argmax) rather than pick one,
to smooth out single-fold false positives — `generate_masks_ensemble.py`
produces `datasets/wound-staging-masks/`, the shared mask dataset every
`exp/` below is built from.

## 3. Shared masks → per-experiment datasets

`datasets/wound-staging-masks/stage_X/{images,masks}/` = the full PIID
staging set with an ensemble-predicted 3-class mask (other/skin/wound) per
image. Two experiment variants built from it so far, both **flat**
(`stage_X/<file>`, no masks saved, matching `wound-stage-classification/`'s
own layout exactly so `task_3/src` can be reused unmodified):

- **exp-1** (`exp/exp1_wound_crop/scripts/generate_dataset.py`): keep
  skin+wound, zero out background. Mean kept-area ≈ 97%.
- **exp-2** (`exp/exp2_wound_dilated/scripts/generate_dataset.py`): keep
  wound only. Problem: 41/1091 images had zero predicted wound pixels, and
  small wounds leave the model almost nothing. Fixed with **adaptive
  dilation**, inversely proportional to original wound size: grow (exact
  circular/Euclidean growth, not a square kernel) until the region covers
  ≥8% of the image or a 70px radius cap is hit; if nothing was detected at
  all, seed a small disk at the image center first, then grow the same way.
  233/1091 images were grown this way (41 of those from a center seed); none
  hit the radius cap.

Both scripts read `datasets/wound-staging-masks/` and never touch the
staging label.

## 4. Experiment training

`exp/exp1_wound_crop/scripts/train.py` — MiT-B1 + CORN (task_3's confirmed
best config, `task_3/results/task_3_outputs/results/ablation_summary.csv`),
trained on `datasets/wound-staging-masks-exp-1/`. Since that dataset has
the exact same layout as the original, `task_3/src/splits.py`'s own
`build_folds()` reproduces the **original PIID 3-fold split** exactly (same
filenames, same seed) — verified by direct comparison before relying on it.
Smoke-tested end to end (checkpoint + results row written correctly), not
yet run for real.

`exp/exp2_wound_dilated/` has its dataset but no `train.py` yet — next
step, same pattern as exp-1 once exp-1's real results are in.

## 5. Checkpoint gaps elsewhere (context for later)

| Model | Status |
|---|---|
| Task 3, MiT-B1+CORN (confirmed best) | metrics exist (`ablation_summary.csv`), but no `.pt` for any fold locally — only needed if a pipeline wants the actual original-RGB-baseline weights, not just its numbers. |
| Task 2, MAnet+MiT-B2 (current best) | no checkpoint at all locally — blocks tissue-composition experiments until retrained or retrieved from Drive. |

(`task_1`–`task_3`'s `checkpoints/`/`results/` are gitignored, so this
reflects what's on this machine, not necessarily Drive.)
