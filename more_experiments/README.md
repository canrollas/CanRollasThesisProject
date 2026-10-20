# more_experiments — Chapter 8: Wound Staging Improvement

Additional experiments for Chapter 8: can staging (Task 3) be improved by
reusing wound localisation (Task 1) and tissue composition (Task 2)
information, without leaking PIID staging data into the models that produce
that information?

**Status: investigation + scaffolding only. No training has been run.**
Everything below is the result of inspecting the datasets and existing
code, per the "immediate instructions" in the project brief (do not train
until this is reviewed and approved).

## 1. PIID overlap — verified

Reproducible via `scripts/verify_piid_overlap.py` (uses
`task_1/src/splits.py`'s existing source-tagging logic, run with
`task_1/.venv/bin/python3 more_experiments/scripts/verify_piid_overlap.py`).
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

**228 is confirmed correct, not 221.** Task 1's own `src/splits.py` already
tags this subset as its own source (`ft_evre1_stage_1` prefix), and all 228
of those filenames match, by trailing numeric id, 228 of the 230 images in
`datasets/wound-stage-classification/stage_1/`. Spot-checked two id matches
by opening the actual image pairs — identical photos (e.g.
`ft_evre1_stage_1_10267.png` ≡ `stage_1_10267.jpg`), confirming this is a
real correspondence, not a coincidence. The other 2 Stage-1 PIID images
(ids `37467`, `98789`) were never in Task 1's training set to begin with.

**Caveat worth keeping in mind:** naive numeric-id matching against
`stage_2/3/4` turns up a handful of *false-positive* collisions (e.g. id
`1221` exists as both `fusc_1221.png` in Task 1 and `stage_3_1221.jpg` in
the classification set), because `fusc`/`wsnet`/`ft_tendonbone` reuse small
integer ids unrelated to PIID. Visually confirmed these are unrelated
photos. Logged in the report so nobody rediscovers them and panics. Only
the `ft_evre1_stage_1` prefix is a real PIID/Task-1 overlap.

## 2. Task 2 (tissue segmentation) — no PIID overlap

No PIID/`evre`/`stage`-pattern filenames exist in
`datasets/wound-tissue-segmentation/tissue_images/`. Its 612 images (491
unique case-uuids, some with multiple wounds) are built from the *same* 488
unique clinical-photo uuids as Task 1's `ft_tendonbone` subset — i.e. **Task
1 and Task 2 share source photos with each other**, but neither shares
anything with PIID staging. No Task-2→Task-3 leakage risk; Task 2 does not
need a leakage-free retrain. (The Task1↔Task2 overlap is a separate,
lower-priority note — both are segmentation tasks on the same clinical
images, not a staging-evaluation leak.)

## 3. Checkpoint availability — gaps found

| Model | Status |
|---|---|
| Task 1, MAnet+MiT-B2, all 3 folds | ✅ present — `task_1/results/task_1_outputs/checkpoints/manet__mit_b2__imagenet_fold{0,1,2}.pt`. **Not reusable here** (trained on PIID-contaminated data) — reference only, e.g. for an "original vs. leakage-free Task 1" sanity comparison. |
| Task 3, MiT-B1+CORN (confirmed best: 81.7% acc, QWK 0.914 in `task_3/results/task_3_outputs/results/ablation_summary.csv`) | ⚠️ **metrics exist, checkpoints do not.** No `.pt` file for `mit_b1__corn` in `task_3/results/task_3_outputs/checkpoints/` for any fold — only CSV results were kept locally. Experiment 1A/2A can reuse the *numbers*, but any pipeline needing the actual weights (e.g. as a fine-tuning start point) requires retraining this config first. |
| Task 2, MAnet+MiT-B2 (current/post-fix best) | ❌ **no checkpoint at all.** `task_2/results/` has qualitative panel PNGs and metric CSVs (so it was trained at some point, likely on Colab) plus a `legacy/checkpoints_pre_fix/` set from *before* the regularization revert (commit `5cb9e6d`) — but no current-config weights locally. **Step/Experiment 2 is blocked on this** until Task 2 is retrained or its Colab/Drive checkpoint is retrieved. |

(All three tasks' `checkpoints/`/`results/` dirs are gitignored — Colab-generated, never committed — so this reflects what exists on *this* machine, not necessarily what exists on Drive.)

## 4. Open decisions before training starts

- **`piid_crops/configs/crop_config.yaml` — `fallback_strategy`**: what happens when the leakage-free Task 1 model finds no wound (expected for some intact-skin Stage 1 photos)? Must be fixed in advance and not depend on staging labels.
- **`task2_tissue_predict/configs/tissue_predict_config.yaml` — `tissue_representation`**: probability maps vs. categorical vs. continuous feature — brief explicitly forbids defaulting to arbitrary ordinal integers.
- **`task2_tissue_predict/configs/tissue_predict_config.yaml` — `no_wound_bed_handling`**: Stage 1 images with intact skin need explicit handling, not a fabricated tissue prediction.
- **`exp1_crop_robustness/configs/perturbations.yaml`**: shift/scale/bbox-noise ranges — must be selected on train/val, never on the staging test fold.
- **Task 2 checkpoint**: retrain, or retrieve from Drive? Blocks all of Experiment 2.

## 5. Folder structure

```
more_experiments/
├── manifests/                    verified PIID overlap report + leakage-free filelists (generated, tracked)
├── scripts/verify_piid_overlap.py   reproduces section 1 from raw files, no cached assumptions
├── task1_retrain/                 Step 1: leakage-free Task 1 (MAnet+MiT-B2) — configs/checkpoints/results
├── piid_crops/                    Step 2: masks/bboxes/crops from the leakage-free model — configs/manifests/crops
├── exp1_wound_crop/                Experiment 1B: staging on crops (MiT-B1+CORN) — configs/checkpoints/results
├── exp1_crop_robustness/           Experiment 1C: eval-only, perturbation sweep over 1B's checkpoint
├── task2_tissue_predict/           tissue predictions for Experiment 2 — configs/predictions
├── exp2_tissue_staging/            Experiment 2B: RGB+tissue staging — configs/checkpoints/results
├── src/                            shared utilities (architectures, metrics) once implementation starts
└── results/                        cross-experiment comparison tables + figures (E1-A..E2-B)
```

`checkpoints/`, `results/`, `crops/`, `predictions/`, and the generated
parts of `manifests/` under each subfolder are gitignored, same convention
as `task_1/`–`task_3/`. Original thesis experiments and checkpoints under
`task_1/`, `task_2/`, `task_3/` are untouched by anything here.

## 6. Step 1 is implemented: leakage-free Task 1 retrain

`task1_retrain/` now has real code, not just a config placeholder:

- `leakagefree_splits.py` — same stratified-by-source 3-fold CV as
  `task_1/src/splits.py`, but sourced from
  `../manifests/task1_leakagefree_filelist.txt` instead of listing the raw
  image folder (which still has the 228 PIID images). Refuses to build
  folds if any `ft_evre1_stage_1` file sneaks into the filelist.
- `run_leakagefree.py` — trains the single MAnet+MiT-B2/ImageNet config for
  3 folds, reusing `task_1/src`'s model/losses/engine/metrics/augmentations
  code verbatim. Same resume-on-epoch-crash behaviour as
  `task_1/run_ablation.py`. Writes to `./checkpoints` and `./results` only
  — `task_1/` itself is untouched.
- `colab_leakagefree_retrain.ipynb` — Colab notebook (mount Drive, clone
  repo, fetch the same `wound-region-segmentation.zip` Task 1 uses,
  re-verify the leakage-free manifest against that copy of the data, redirect
  checkpoints/results to Drive, train, preview samples).

**Smoke-tested locally** (`--smoke`: 1 fold, 2 epochs, 8/4/4 train/val/test
images) end-to-end on this machine — correctly builds the leakage-free
3-fold split (train/val/test sizes ≈ 1948/217/1083 before the smoke
subsample), trains, and writes a checkpoint + results row + sample panel.
Artifacts from that smoke run were deleted afterward (not real results).

### Running on Colab

1. Push this repo (`more_experiments/` is all text — configs, scripts, two
   `.py` files, one notebook; no binaries) to GitHub.
2. Open `task1_retrain/colab_leakagefree_retrain.ipynb` in Colab, pick a GPU
   runtime, run cells top to bottom. Needs
   `drive/MyDrive/thesis/wound-region-segmentation.zip` to already exist
   (same zip Task 1's own notebook uses).
3. Outputs land in `drive/MyDrive/thesis/task1_leakagefree_outputs/` —
   separate from Task 1's own `task_1_outputs/`, so the original checkpoints
   are never at risk of being overwritten.

This is still only **Step 1** (MAnet+MiT-B2, 1 architecture, as requested —
not the full original ablation grid). Steps 2–3 (crop generation, staging
retrain) still need their own training code before they can run.
