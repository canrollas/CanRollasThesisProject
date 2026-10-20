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
  wound only, adaptively grown. Two problems with a naive version: (a) a
  hard cutoff meant already-large wounds got zero surrounding skin at all;
  (b) growing small/undetected wounds by exact circular distance
  thresholding produced a suspiciously clean, perfectly round blob -- an
  artifact a model could learn to key on instead of actual wound shape.
  Fixed with a `--base-margin-px` skin ring applied to *every* image
  unconditionally (so there's no cliff at the target threshold), plus any
  further growth (for small/undetected wounds, up to a target coverage or a
  70px radius cap) perturbed by smooth low-frequency noise so the boundary
  reads as organic instead of geometric. 41/1091 images had zero predicted
  wound pixels and seed from the image center; 124/1091 got extra noisy
  growth beyond the base margin; none hit the radius cap.

Both scripts read `datasets/wound-staging-masks/` and never touch the
staging label.

## 4. Experiment training — results

`exp/exp1_wound_crop/scripts/train.py` and `exp/exp2_wound_dilated/scripts/train.py`
— MiT-B1 + CORN (task_3's confirmed best config), trained on
`datasets/wound-staging-masks-exp-{1,2}/`. Both reuse `task_3/src` wholesale
(model/losses/engine/metrics/augmentations/splits/utils) — only the data
content differs. Since both datasets have the exact same layout as the
original `wound-stage-classification/`, `task_3/src/splits.py`'s own
`build_folds()` reproduces the **original PIID 3-fold split** exactly (same
filenames, same seed) — verified by direct comparison, not assumed. Trained
on Colab, `--batch 16` (matching the baseline's `configs/base.yaml`
default — an earlier run used `--batch 32`, caught and corrected before
trusting these numbers, since half as many gradient steps/epoch at 100
epochs is a real confound, not noise).

| Metric | **E1-A** original RGB (`task_3` baseline) | **E1-B** exp-1 (skin+wound) | **E2** exp-2 (wound only, dilated) |
|---|---|---|---|
| Accuracy | 0.8168 ± 0.0150 | 0.7819 ± 0.0157 | 0.7837 ± 0.0222 |
| Macro F1 | 0.8182 ± 0.0138 | 0.7815 ± 0.0149 | 0.7863 ± 0.0214 |
| QWK      | 0.9138 ± 0.0164 | 0.8935 ± 0.0111 | 0.9000 ± 0.0149 |

(mean ± std over 3 folds; full per-fold rows in each `exp/.../results/*.csv`)

**Finding: both masking variants underperform the original full-frame RGB
baseline**, consistently, by a similar margin (~3.3pp accuracy, ~1.4–2.0pp
QWK) — the opposite of the hypothesis that giving the model an explicit
wound-localisation crop would help staging. E1-B and E2 are statistically
indistinguishable from *each other* (within 1 std both ways) despite
keeping very different amounts of context (skin+wound vs. wound only) — so
it isn't "how much you mask", masking at all costs something the model was
using. Plausible explanation: the full photo's background/framing/clinical
context (bed linens, lighting, scale cues) was contributing real signal
that a flat black mask removes, more than localisation was adding.

Not yet re-run with `--batch 32` repeated as a sanity check that the
original (unfixed) exp-1 number (0.7782 acc) was close to the `--batch 16`
one (0.7819) purely by chance rather than batch size mattering more than
this single comparison suggests — the two are close enough (within 1 std)
that this wasn't pursued further, but worth knowing if these numbers get
revisited.

## 5. Checkpoint gaps elsewhere (context for later)

| Model | Status |
|---|---|
| Task 3, MiT-B1+CORN (confirmed best) | metrics exist (`ablation_summary.csv`), but no `.pt` for any fold locally — only needed if a pipeline wants the actual original-RGB-baseline weights, not just its numbers. |
| Task 2, MAnet+MiT-B2 | checkpoints ARE present and usable: `task_2/results/task_2_outputs/checkpoints/wound_inner_fold{1,2,3}/MAnet_mit_b2_imagenet.pth`. Not a gap — use these directly for tissue-composition experiments (exp-3 etc). |

(`task_1`–`task_3`'s `checkpoints/`/`results/` are gitignored, so this
reflects what's on this machine, not necessarily Drive.)

## 6. exp-3 — RGB + tissue-probability channels

`exp/exp3_tissue_channels/` — 8-channel input (full uncropped RGB, channels
0-2, + 5 per-pixel tissue-class probabilities from Task 2, channels 3-7:
granulation/slough/necrosis/tendon/bone), same MiT-B1+CORN protocol and
verified-identical PIID 3-fold split as exp-1/exp-2.

**First-layer init is explicit, not SMP's automatic `in_channels=8`
mechanism** — checked what that automatic path actually does to MiT-B1's
first conv and it was wrong for this ablation: it cyclically tiles the
pretrained 3-channel weights across all 8 channels *and* scales every
channel (RGB included) by 3/8, so even the RGB pathway starts from
different weights than the true RGB-only baseline. Replaced with an
explicit expansion (`scripts/model.py`): RGB channels copied unchanged
from the pretrained conv, tissue channels zero-initialized — the
8-channel model is mathematically identical to the RGB-only baseline at
initialization (verified directly: `forward(RGB, tissue=0) ==
forward_3ch(RGB)` exactly), and any divergence after training is then
attributable to the model learning to use tissue information, not a
different starting point. `scripts/test_8ch_model.py` checks this
property plus gradient flow into the zero-initialized channels before any
training is trusted — run it first.

Data provenance: `datasets/wound-staging-masks-exp-3/provenance.json`
records the Task 1/Task 2 checkpoints used, channel order, and the
leakage check (Task 2's training data has zero overlap with the PIID
staging images these tissue predictions are generated for — verified in
section 1 above, not assumed for exp-3). Each training checkpoint also
gets a `.provenance.json` sidecar (config, pretrained-weight source,
first-conv init method).

Not yet trained for real (smoke-tested only) — `colab_train.ipynb` ready,
dataset zipped (`datasets/wound-staging-masks-exp-3.zip`, ~144MB) for
Drive upload.
