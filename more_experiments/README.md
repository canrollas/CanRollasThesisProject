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

**Trained for real** (Colab, corrected zero-init, `batch_size: 16`):

| Metric | E1-A baseline | E1-B exp-1 | E2 exp-2 | **E3 exp-3** |
|---|---|---|---|---|
| Accuracy | 0.8168 | 0.7819 | 0.7837 | **0.7791** |
| Macro F1 | 0.8182 | 0.7815 | 0.7863 | **0.7822** |
| QWK      | 0.9138 | 0.8935 | 0.9000 | **0.8997** |

Same pattern as exp-1/exp-2: still below the full-frame RGB baseline, by
a similar margin — concatenating tissue-probability channels at the input
didn't help, numerically close to exp-2's own result. One more data point
toward the same conclusion (section 4): adding this derived information
doesn't straightforwardly improve staging, regardless of which of the
three ways tried so far (background-mask crop, wound-only crop, early
fusion with tissue channels) it's added. exp-4's late fusion is the next
test of whether *how* it's combined (not just *what's* added) matters.

## 7. exp-4 — late fusion (RGB MiT-B1 + tissue CNN)

`exp/exp4_late_fusion/` — tests the opposite hypothesis from exp-3: RGB
and tissue information encoded by **separate** networks, combined only
after each produces its own pooled feature vector, instead of
concatenated at the input and run through one shared conv stack.

- RGB arm: `task_3/src/model.py`'s own `Backbone` (ImageNet MiT-B1),
  unchanged — `F_rgb = GAP(MiT-B1(rgb))`, 512-dim.
- Tissue arm: a small 3-layer CNN (`scripts/model.py`'s `TissueCNN`) —
  `F_tissue = GAP(CNN(tissue_5ch))`, 128-dim.
- Fusion (`--fusion concat|gated`):
  - `concat`: `Z = MLP([F_rgb; F_tissue])` (640→256→256), `CornHead(256)`.
  - `gated`: `g = sigmoid(MLP_g([F_rgb; F_tissue]))`,
    `F_fused = F_rgb + g ⊙ (W · F_tissue)`, `CornHead(512)` directly on
    `F_fused` — lets the model learn, per sample, how much to trust a
    given tissue prediction instead of using it at fixed strength. Only
    worth running if `concat` beats the baseline/exp-3 first.
- Optional staged freeze→fine-tune (`--freeze-epochs N`, off by default):
  freezes the RGB backbone for the first N epochs (only the tissue arm +
  fusion head train), then unfreezes and continues at a lower
  `finetune_lr`. Off by default so the first concat-vs-baseline-vs-exp-3
  comparison uses the same single-phase training protocol as every other
  experiment here — not an extra confound on top of the fusion question.
- **RGB arm starts from plain ImageNet weights**, not a staging-finetuned
  checkpoint — task_3's own trained MiT-B1+CORN baseline has no saved
  `.pt` locally (section 5), so there's nothing to initialize from yet.
  Documented, not silently assumed away.

Reuses exp-3's exact dataset (`datasets/wound-staging-masks-exp-3/`,
`scripts/dataset.py` splits each `.npy` back into separate RGB/tissue
tensors instead of concatenating) — same leakage-safe tissue predictions,
same verified PIID split, no new data generation or leakage surface.
`scripts/test_late_fusion_model.py` checks both fusion modes' shapes,
finite loss, gradient flow into the tissue CNN, gradient flow into the
RGB backbone by default, and that freezing the RGB backbone actually
stops its gradients — all pass. Smoke-tested end to end (concat, gated,
and gated+freeze all run cleanly); not yet trained for real.

**Trained for real, both fusion modes** (RGB arm still ImageNet-init, not
baseline-init — see exp-5 below for why that matters):

| Metric | E1-A baseline | E3 exp-3 (early fusion) | **E4-concat** | **E4-gated** |
|---|---|---|---|---|
| Accuracy | 0.8168 | 0.7791 | **0.7983** | 0.7957 |
| Macro F1 | 0.8182 | 0.7822 | **0.8003** | 0.7977 |
| QWK      | 0.9138 | 0.8997 | **0.9109** | 0.9063 |

Late fusion clearly beats early fusion (exp-3) and exp-1/exp-2 on every
metric, and closes most of the gap to the baseline — QWK in particular
(0.9109 vs. 0.9138) is within 1 std of the baseline's own std (0.0164).
Still short of the baseline on accuracy/macro F1 though, and `concat`
slightly edges out `gated` here (gated's higher std, 0.0238 vs 0.0156,
suggests the extra gating complexity added variance without a
compensating accuracy gain on this data size — not yet enough evidence to
call `gated` the better choice). First real signal in this chapter that
*how* RGB and tissue information are combined matters as much as *whether*
it's added at all.

## 8. exp-5 — residual late fusion (baseline-preserving)

`exp/exp5_residual_fusion/` — `z_final = z_rgb + delta_z`. `z_rgb` comes
from the **frozen, trained RGB MiT-B1+CORN baseline** loaded per fold
(unlike exp-4, whose RGB arm was ImageNet-only — no trained baseline
checkpoint existed locally at the time). `delta_z` comes from exp-4's
`TissueCNN` + a small residual MLP whose final linear layer is
zero-initialized, so at construction `delta_z = 0` and the model
reproduces the baseline's predictions exactly — verified two ways before
any training is trusted: `scripts/test_residual_model.py`
(architecture-level, 11 checks, no checkpoint needed) and, in
`scripts/train.py` itself, against the real loaded checkpoint and real
data (fails loudly and stops if the fusion model's `z_rgb` doesn't match
a reference `task_3/src/model.py` instance with the same weights, byte
for byte). Phase 1 trains only the tissue CNN + residual head — the RGB
branch (backbone + CORN head) is frozen for the whole run, with real
early stopping (not just best-checkpoint selection) on validation
macro-F1.

**Blocked on the RGB baseline checkpoint — now partially unblocked.**
`task_3`'s own trained `mit_b1__corn` checkpoints were never saved
locally (section 5's long-standing gap). Fold 1's was provided directly
and placed at
`task_3/results/task_3_outputs/checkpoints/mit_b1__corn_fold1.pt`; folds
0 and 2 are still missing. `scripts/train.py` fails loudly with the exact
`task_3/run_ablation.py --configs mit_b1__corn` command to fix it if a
fold's checkpoint isn't found, rather than silently substituting ImageNet
init — this is NOT a re-ablation, just producing the already-characterized
config's weights, which happen to never have been saved. `colab_train.ipynb`
automates this: checks `drive/MyDrive/thesis/task_3_outputs/` for existing
fold checkpoints first, trains only whichever are still missing.

**Fold 1 result (real, converged, not a smoke test):**

| Metric | Fold 1's own baseline | **exp-5 fold 1 (residual fusion)** |
|---|---|---|
| Accuracy | 0.8370 | **0.8232** |
| Macro F1 | 0.8369 | **0.8244** |
| QWK      | 0.9332 | **0.9262** |

Closer to its baseline than either exp-4 variant got to the overall
baseline mean — early-stopped at epoch 12 (best at epoch 2).

**3-fold result, `rgb_tissue` variant (complete):** accuracy 0.8158 ±
0.0073, macro F1 0.8178 ± 0.0073, QWK 0.9132 ± 0.0122 — statistically
indistinguishable from the baseline (0.8168/0.8182/0.9138), and with a
*smaller* std than the baseline's own across all three metrics.

**Root-cause diagnostic (`scripts/diagnose_residual.py`) found this isn't
really "matching the baseline via genuine tissue correction"**: every
fold's best checkpoint was saved from epoch 1–2 (near-zero `delta_z`);
later epochs grew `delta_z`'s norm steadily while val macro-F1 *degraded*
(classic overfit-then-rejected-by-early-stopping). At the converged
checkpoint, `delta_z`'s magnitude is ~1000x smaller than `z_rgb`'s
decision margins, the tissue CNN's gradient is ~1e-3–5e-3 (near-flat), and
— most tellingly — swapping in a **different image's tissue map**
(shuffled, same RGB) changes `delta_z` by cosine ≈ 1.0 (same direction
regardless of content). The branch isn't using tissue information; it's
converging to a near-baseline no-op.

**Architecture fix, `tissue_only` variant** (`config_id:
mit_b1__corn__tissue_only_residual`): `delta_z` computed from tissue CNN
features ONLY (`TissueOnlyResidualHead`, `scripts/model.py`) —
structurally cannot see RGB, verified directly
(`scripts/test_residual_model.py`: changing RGB with tissue fixed changes
`z_rgb` but leaves `delta_z` byte-identical). Also: GroupNorm instead of
BatchNorm in the tissue CNN (batch_size=16 made BatchNorm's running stats
noisy), dropout=0.2 in the residual head. LR unchanged (1e-4) — one
isolated change at a time.

Fold 1 result (only fold trained so far): accuracy 0.8370, macro F1
0.8376, QWK 0.9323 — val-F1 now improves *monotonically* for the first 2
epochs (0.8232→0.8574, vs. `rgb_tissue`'s immediate peak-then-decay),
beats `rgb_tissue`'s fold 1 result on every metric, and matches/slightly
exceeds the baseline's own fold 1. **But the tissue-ablation diagnostic
still shows near-zero content-dependence** (cosine(real,shuffled)≈1.0,
relative L2(real,shuffled)/‖delta_z‖≈0.005, tissue CNN gradient still
~1e-3) — the architecture fix removed RGB as an escape route but hasn't
yet given the tissue branch a strong enough learning signal to actually
discriminate tissue content; the fold 1 improvement is plausibly a
near-constant shift that happens to help a few borderline predictions,
not genuine tissue-driven correction. Lowering the LR further (as
originally planned as a follow-up) would likely make this worse, not
better, given the gradient is already this weak — an open question, not
yet resolved. **Folds 0 and 2 for `tissue_only` still needed.**

## 9. exp-6 — hand-crafted tissue statistics (isolating the signal)

`exp/exp6_tissue_statistics/` — steps back from a *learned* tissue
encoder (exp-5's CNN, diagnosed with a weak/near-flat gradient) to
simple, deterministic, hand-computed statistics, to answer a narrower
question first: do the tissue probability maps carry *any* independent
staging signal at all, before trying to extract it better.

**Features** (`scripts/extract_tissue_features.py`): for each of the 5
tissue classes, mean / std / 95th-percentile / fraction-above-0.5 of the
predicted probability, computed **only over predicted wound pixels**
(the same leakage-free Task 1 ensemble mask exp-3 used to gate the
channels in the first place — never ground truth) → 20-dim vector per
image. 41/1091 images with zero predicted wound pixels get an all-zero
vector (same deterministic-missing-feature convention as exp-3/5, not a
fabricated value). Identical preprocessing for every split — nothing
computed per-fold or train/test-specific.

### tissue_statistics_only — is there a signal at all?

MLP+CORN on the 20-dim vector **alone, no RGB anywhere**
(`scripts/train_stats_only.py`, reuses `task_3/src/engine.py` unchanged).
3-fold result: **accuracy 0.3803 ± 0.0094, macro F1 0.3065 ± 0.0151, QWK
0.3550 ± 0.0091**. Far below the RGB baseline (0.8168/0.8182/0.9138) —
but clearly **above chance** (~0.25 accuracy, ~0 QWK for 4 roughly-balanced
classes with no signal). **Answer: yes, tissue composition alone carries
real, non-trivial independent staging signal — just much weaker than
RGB appearance.** This is consistent with every fusion experiment so far
only managing small corrections: there's a real but modest amount of
extra signal to extract, not a large one being missed.

### tissue_statistics_residual_fusion — fusing it in

Same `z_final = z_rgb + delta_z` mechanism as exp-5, residual MLP on the
20-dim vector instead of CNN features (`scripts/train_residual.py`,
reuses `exp5/scripts/engine.py` unchanged). All 3 folds now trained
(fold0/fold2 RGB baseline checkpoints obtained and copied in locally,
no Colab needed): **accuracy 0.8159 ± 0.0138, macro F1 0.8175 ± 0.0129,
QWK 0.9131 ± 0.0156** — per-fold: fold0 0.8122/0.8135/0.9152, fold1
0.8343/0.8350/0.9310, fold2 0.8011/0.8041/0.8930. Essentially
indistinguishable from the RGB baseline (0.8168 ± 0.0150 / 0.8182 ± 0.0138
/ 0.9138 ± 0.0164) and from exp-5's best CNN-tissue result — the mean
sits a hair below baseline, well inside the fold-to-fold noise.

**Tissue-shuffle ablation** (`scripts/diagnose_residual_stats.py`, same
method as exp-5's, run on all 3 folds): cosine(real, shuffled) ≈ 1.0 in
every fold, but **relative L2(real,shuffled)/‖delta_z‖ = 0.076 / 0.140 /
0.154** (fold0/1/2, mean ≈ 0.12) — roughly **25x higher** than the
CNN-tissue variant's 0.0045. Swapping in a different image's tissue
statistics now visibly moves the correction's magnitude in every fold,
not just leaving it unchanged. Bypassing the CNN's weak-gradient
bottleneck with deterministic features measurably increased
content-sensitivity — a genuine, if partial, step forward on the open
question from exp-5. **But the increased content-sensitivity did not
convert into an accuracy gain**: fold1 is the only fold where the
residual actually flips any predictions (2/73, both fixes, pushing that
fold to 0.8630 vs RGB-only 0.8356 on the validation split used for
diagnosis), while fold0 and fold2 change zero predictions. **Conclusion:
the hand-crafted tissue signal is real and more extractable than the CNN's,
but still too weak/noisy relative to the strong RGB baseline to move
aggregate 3-fold accuracy — consistent with `tissue_statistics_only`'s
own ~0.38 standalone accuracy being far below the ~0.82 RGB baseline.**

### Which tissue features actually discriminate stage? (`scripts/analyze_tissue_stats.py`)

One-way ANOVA per feature across Stage 1–4 (1,050 images with a detected
wound; `tissue_statistics_anova.csv` has the full table). 17/20 features
are significant (p<0.05), 14 at p<0.001 — consistent with the
above-chance `tissue_statistics_only` result. Most discriminative:

| Feature | F-stat | Stage 1→4 mean |
|---|---|---|
| `slough_std` | 146.1 | 0.225 → 0.310 → 0.370 → 0.368 |
| `granulation_std` | 103.2 | 0.218 → 0.314 → 0.354 → 0.360 |
| `slough_p95` | 72.1 | 0.755 → 0.876 → 0.982 → 0.970 |
| `necrosis_std` | 30.4 | 0.060 → 0.061 → 0.127 → 0.121 |
| `granulation_mean` | 28.0 | 0.509 → 0.535 → 0.369 → 0.431 |

Within-wound *heterogeneity* of slough/granulation probability (the std
features) is far more discriminative than their mean — clinically
plausible (a more severe wound bed is more heterogeneous: mixed
granulation/slough/necrosis rather than one uniform tissue). Necrosis
std/p95 jump specifically at stage 3→4, matching the expected clinical
picture. One caveat: `bone_mean` is statistically significant but with
means essentially reversed from clinical expectation (highest at stage 1,
near-zero elsewhere) while `bone_frac_above_0.5` is NOT significant
(p=0.41) — consistent with this being a handful of noisy, low-confidence
false-positive pixels (bone is Task 2's rarest class) rather than a real
effect; not treated as a genuine finding. `tendon`/`bone`
`frac_above_0.5` are the only features that don't reach significance at
all — both rare classes with little stable per-image signal to measure.

## 10. exp-7 — global–local ROI feature fusion (MiT-B1 + CORN)

`exp/exp7_global_local_roi_fusion/` — adapts the global–local fusion
architecture from Bayraktar et al. (2026, *Integrating Localization,
Feature Fusion, and Attention for Pressure Ulcer Staging from Clinical
Images*, JIIM, Fig. 4 — not the attention-enhanced Fig. 5 model). No
tissue segmentation, depth estimation, extra large encoder, or attention
anywhere in this experiment, per the brief.

**Architecture.** `logits = fusion_head(concat(F_global, F_local))`.
`F_global` = GAP of the frozen, trained RGB MiT-B1+CORN baseline's
final-stage feature map (`[B,512,7,7] → [B,512]`, verified empirically,
not assumed). `F_local` = `roi_align` over the **third** MiT-B1 stage
(`[B,320,14,14]`, higher spatial resolution than the final stage, also
verified empirically) at the predicted wound box, spatially average-pooled
and projected to 128 dims. `fusion_head` is the baseline's own CornHead
**widened** from 512→640 inputs: the 512 global columns + bias copied
exactly, the 128 local columns zero-initialized — `fusion_logits ==
baseline_logits` **exactly** at construction (`torch.equal`, 12/12
architecture tests pass in `scripts/test_model.py`, re-verified against
the real checkpoint in `scripts/train.py` before every training run).
Phase A: MiT-B1 backbone frozen (verified via an unchanged-`f_global`
probe-batch check after training); only `local_proj` + `fusion_head`
train (0.043M / 13.194M params). Image/box pairs stay geometrically in
sync under the existing hflip/vflip/rotate augmentation policy via a new
`PairedImageBoxAugmentation` (box-corner transform formulas derived and
empirically verified against `np.rot90`'s actual pixel mapping — see
`scripts/augmentations.py`).

**ROI generation** (`scripts/generate_roi_manifest.py`) reuses the
already-generated leakage-free Task 1 ensemble masks
(`datasets/wound-staging-masks/`, from `piid_crops/generate_masks_ensemble.py`
— no staging labels or ground-truth masks used). Fixed rule: largest
connected wound-class component → bbox in original coordinates → scaled
independently per axis into the 224×224 staging coordinate system → +8px
margin. 41/1091 images with zero predicted wound pixels (all stage_1 —
confirmed, not assumed) get the full-image box as fallback, never
excluded. `results/roi_manifest.csv`, 1091/1091 rows, 0 degenerate boxes.

**Results (local sanity run, 3 folds, all baseline-preservation checks
passed, backbone-frozen check confirmed `True` every fold):**

| Metric | RGB baseline | exp-7 fusion |
|---|---|---|
| Accuracy | 0.8168 ± 0.0150 | 0.8120 ± 0.0098 |
| Macro-F1 | 0.8182 ± 0.0138 | 0.8142 ± 0.0095 |
| QWK | 0.9138 ± 0.0164 | 0.9131 ± 0.0090 |

Per-fold: acc 0.7983/0.8204/0.8174, F1 0.8008/0.8224/0.8193, QWK
0.9114/0.9249/0.9030. Essentially indistinguishable from the baseline —
inside fold-to-fold noise, consistent with every other fusion attempt in
this chapter (exp-3/4/5/6).

**Step 8 — ROI contribution ablation** (`scripts/roi_ablation.py`,
inference-only, same trained weights): condition A (predicted ROI) vs. B
(full-image ROI) vs. C (shuffled ROI from a different image, fixed
derangement, no self-pairs). **Zero prediction flips in all 3 folds, all
3 conditions** — accuracy/F1/MAE/QWK identical to 4 decimal places
regardless of which box is used. Per the brief's explicit criterion, this
means **the model did not learn to use wound-specific localization
information** — not "similar performance", literally identical
predictions. Root cause confirmed directly (not assumed): the
fusion-head's 128 local weight columns did move away from zero during
training (abs-mean ≈0.0006–0.001) but stayed ~20–35× smaller than the
512 global columns (abs-mean ≈0.023) — small enough that no image's
local-feature contribution ever crosses a CORN decision boundary. Early
stopping triggered at epoch 11 in every fold (patience=10), consistent
with the best validation score being found almost immediately and never
improved on — the same "near-zero-correction checkpoint saved at epoch
1–2" pattern diagnosed in exp-5's original residual design.
**Limitation, reported as required:** condition C's shuffled boxes still
overlap the receiving image's own real box surprisingly often (IoU>0 for
353–359/362–367 per fold, mean IoU ≈0.40–0.44) — expected, since wound
photos are usually center-framed at similar scale — but since real,
full-image, and shuffled conditions are all bit-identical in their
predictions anyway, this limitation doesn't change the conclusion.

**Step 9 — evaluation and error analysis** (`scripts/error_analysis.py`):
per-class F1 ranges from 0.66 (stage_3, fold0) to 0.91 (stage_1, fold1);
confusion matrices show the dominant error is stage_3↔stage_4 (and to a
lesser extent stage_1↔stage_2), matching the baseline's own known
confusion pattern (unsurprising, since fusion predictions are identical
to the global-only pathway almost everywhere). Paired comparison vs. RGB
baseline, pooled across folds (n=1091): both correct 871 (79.8%), RGB-only
correct 20 (1.8%), fusion-only correct 15 (1.4%), both wrong 185 (17.0%) —
a near-wash, 5 fewer net corrections than losses. Failed-localization
subset (41 images, all stage_1): 100% accuracy in every fold — not a
sign the model handles localization failure well, but a direct
consequence of (a) failures being exclusively "easy", already-intact-skin
stage_1 cases and (b) fusion predictions being identical to RGB-only's
own (already-strong) stage_1 performance.

**Conclusion:** the baseline-preserving zero-init mechanism and the ROI
pipeline are both independently verified correct (architecture tests,
real-checkpoint re-verification, frozen-backbone probe, box-transform
empirical check) — this is not a plumbing failure. The *architecture*
itself, at least under Phase A (frozen backbone, 0.043M trainable
params, early-stopped at epoch 11), does not extract a usable
localization-specific signal from the frozen MiT-B1 features: the global
pathway already dominates the fusion head so completely that the local
branch's learned weights stay below the threshold needed to ever change
a prediction. This is a genuine negative result for Research Question 7,
not an inconclusive one — the strongest and cleanest "no" among all of
Chapter 8's fusion attempts (0 flips vs. small-but-nonzero flips/
corrections in exp-3/4/5/6). Phase B (unfreezing the last MiT-B1 stage,
`scripts/model.py`'s `unfreeze_last_stage()`, already implemented and
tested) was not run, per the brief's "do not run additional architectural
searches before evaluating Phase A" instruction.

A Colab notebook (`colab_train.ipynb`) is provided to reproduce this run
for the official record (GPU timing, not a different experiment) and to
re-run Steps 8–9 against those checkpoints.

**Scope of this negative result — exactly what is and isn't claimed:**

1. Global–Local ROI Fusion did **not** beat the RGB baseline (0.8120 vs.
   0.8168 accuracy, 0.8142 vs. 0.8182 macro-F1 — both inside the
   baseline's own fold-to-fold noise, not a regression either).
2. In the predicted/full-image/shuffled ROI ablation, **no single
   per-image class prediction changed** across all 3 folds (0 flips) —
   verified directly, not inferred from aggregate metrics alone.
3. This result is scoped **specifically to the tested Phase A
   architecture** — MiT-B1 backbone frozen, only `local_proj` +
   `fusion_head` (0.043M params) trained, early-stopped at epoch 11 in
   every fold.
4. It does **not** establish that wound-localization information is
   generally unhelpful for staging — only that *this* frozen-backbone,
   zero-init-widened-head fusion design failed to extract a usable
   signal from it. A different fusion mechanism, a different feature
   stage, or allowing the backbone itself to adapt could behave
   differently.
5. **Joint fine-tuning (Phase B) was not run.** `unfreeze_last_stage()`
   is implemented and unit-tested (`scripts/test_model.py`) but Phase B
   training was never executed, per the brief's instruction not to run
   additional architectural searches before Phase A was fully evaluated.

**Status: Phase A complete.** Checkpoints, ROI manifest, ablation/error-
analysis results, and this write-up are the final, preserved record for
Chapter 8 reporting. No Phase B or new fusion architecture is queued to
start automatically.

## 11. exp-10 — joint tissue × relative-geometry ordinal tokens

[`exp/exp10_tissue_geometry_tokens/README.md`](exp/exp10_tissue_geometry_tokens/README.md)
documents a new candidate architecture motivated by the inactive auxiliary
branches in exp5/7 and the tissue-heterogeneity signal in exp6. It pools
spatial RGB feature means and variances into soft tissue × relative-depth
tokens, then uses three ordinal queries with direct auxiliary supervision
and joint last-stage fine-tuning. Matched RGB, tissue, wound-geometry and
full controls are implemented. Raw Depth Anything V2 Small maps were
generated and verified for all 1,091 images.

**Status: implementation and small pipeline checks complete; no full
three-fold training/test result yet. No improvement over the 0.8168 RGB
baseline is claimed.** RGB-derived pseudo-depth is a pretrained geometric
prior, not measured physical wound depth. The experiment includes separate
validation/test evaluation, input/split/checkpoint identity checks and
content-shuffle ablations; the historical exp1–7 records remain unchanged.
