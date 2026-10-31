# Exp-14 — RGB + Monocular Relative-Depth Channel

**Status:** scaffolded, unit-tested, local smoke test passed. No 3-fold
result yet — **0.8168 is not claimed to be beaten**. This is the first
experiment in Chapter 8 that actually trains a network on RGB+depth;
every earlier depth-related result (WP5,
`more_experiments/experiment_extended/wp5_depth_stage34/`) was a hand-fit
threshold rule on depth summary statistics, explicitly scoped as "not a
full depth-fusion rerun" — this is that rerun.

## What this tests

Does adding Depth Anything V2 Small's per-image monocular relative-depth
map as a 4th input channel (early fusion by concatenation, same mechanism
as exp-3/exp-3.1's tissue channel) give the RGB MiT-B1+CORN baseline any
staging signal that the WP5 rule-based check couldn't extract?

## Why no Task 1/Task 2 dependency

Unlike exp-1/2 (Task 1 wound mask) or exp-3/3.1 (Task 2 tissue
probabilities inside a predicted wound box), the depth map already
covers the **full, unmasked** photo and was already generated for WP5
(`datasets/wound-stage-depth-anything-v2-small/`). Every one of the 1,091
PIID RGB images is pixel-dimension-identical to its depth map (verified
directly over all 1,091 pairs, not assumed), so no crop/resize/bbox step
is needed — `scripts/generate_dataset.py` just loads both and stacks
them.

## Pipeline

1. `scripts/generate_dataset.py` — per image: load RGB jpeg, load the
   matching Depth-Anything-V2-Small `.npy`, per-image min-max normalize
   the depth map to `[0, 255]` uint8 (depth is a *relative*,
   scale/shift-ambiguous geometric prior — see
   `depth_source_dir/provenance.json`'s own "interpretation" field — so
   only within-image ordering is meaningful, not absolute values), stack
   into a `(224, 224, 4)` uint8 `.npy`. No GPU needed; runs locally in
   seconds. Output: `datasets/wound-staging-rgbd-exp-14/`.
2. `scripts/model.py` — same explicit first-conv-expansion mechanism as
   exp-3/exp-3.1: pretrained RGB weights copied unchanged into channels
   0-2, depth channel 3 zero-initialized, so the 4-channel model is
   mathematically identical to the RGB-only baseline at initialization.
3. `scripts/test_4ch_model.py` — verifies that zero-init property, exact
   forward-pass equivalence at init, and non-zero gradient flow into the
   depth channel. Run this before trusting any training result.
4. `scripts/train.py` — MiT-B1 + CORN, 3-fold, same `build_folds(seed=42)`
   split as task_3's own ablation (paired comparison).
5. `colab_train.ipynb` — Colab runner: mount Drive, pull the zipped
   dataset, install `task_3/requirements.txt`, run the sanity checks,
   train 3 folds, report results against the RGB baseline.

## Running it

```bash
# 1. generate the dataset locally (fast, no GPU)
cd more_experiments/exp/exp14_depth_channel/scripts
python generate_dataset.py

# 2. sanity-check the model
python test_4ch_model.py

# 3. zip for Colab upload to drive/MyDrive/thesis/wound-staging-rgbd-exp-14.zip
cd ../../../../datasets
zip -r wound-staging-rgbd-exp-14.zip wound-staging-rgbd-exp-14

# 4. run colab_train.ipynb on a GPU runtime
```

## Reference — RGB baseline (what this is compared against)

| Metric | Mean ± Std |
|---|---|
| Accuracy | 0.8168 ± 0.0150 |
| Macro-F1 | 0.8182 ± 0.0138 |
| MAE | 0.1896 ± 0.0204 |
| QWK | 0.9138 ± 0.0164 |

(`task_3/results/task_3_outputs/results/ablation_summary.csv`, `mit_b1__corn` row.)
