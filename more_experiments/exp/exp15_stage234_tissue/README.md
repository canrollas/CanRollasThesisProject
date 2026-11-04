# Exp-15 — Stage 2/3/4: RGB versus RGB + tissue

This experiment tests the advisor's question: **does predicted tissue information
help discriminate Stages 2, 3, and 4 when Stage 1 is excluded from training?**
It trains new three-class models; it does not reuse four-class staging predictions
as a substitute for three-class training.

## Matched comparison

| Variant | Input | Output |
|---|---|---|
| `rgb` | RGB channels of the existing exp-3 arrays | Stage 2 / 3 / 4 |
| `rgb_tissue` | The same RGB pixels plus five predicted tissue probability maps; exp-4 late fusion / concat | Stage 2 / 3 / 4 |

Both models start from ImageNet MiT-B1 weights and train with a new CORN head
containing two output logits. They use the same image size, augmentation,
100-epoch schedule, batch size 16, optimizer settings, validation criterion,
and train/validation/test lists. Tissue maps come from the existing exp-3 data;
no new tissue segmentation or ground-truth tissue annotation is required.

The RGB input is taken from the same `.npy` files in both variants. This avoids
introducing a different image decoding or preprocessing source into the comparison.
The fusion architecture adds a small tissue CNN and the existing concat head;
the RGB backbone is trainable from the first epoch.

## Preserve the original folds

The complete four-stage dataset is required to reconstruct the original Task 3
splits. The runner first builds those original splits, then removes Stage 1
**separately from each train, validation, and test list**, and only then remaps
Stage 2/3/4 to labels 0/1/2. It does not rebuild a new split on a three-class
directory or simply filter the test set of an old four-class model.

There are 861 eligible images in total: 313 Stage 2, 275 Stage 3, and 273 Stage 4.
Model selection uses validation macro-F1; test metrics use exactly three labels.
Use this experiment's newly trained `rgb` model as the comparator. The old
four-class RGB score evaluates a different task and is not a direct baseline.

## Run in Colab

1. Open `colab_train.ipynb` in Google Colab and select a GPU runtime.
2. Keep the existing `wound-staging-masks-exp-3.zip` at
   `MyDrive/thesis/wound-staging-masks-exp-3.zip`, or edit `ZIP_PATH` in the
   notebook. Nested archive directories are detected automatically.
3. Run the cells in order. The downloadable notebook includes the experiment
   sources, so it does not require the new code to have been published to GitHub.
4. Run the short smoke test, then the full two-model, three-fold comparison.
5. Use the final cell to download the results ZIP.

Defaults persist outputs under
`MyDrive/thesis/exp15_stage234_tissue_outputs/`. A disconnected session can
continue with the same notebook and output directory: completed model/fold runs
are skipped and an unfinished run resumes from its saved epoch. Smoke outputs
are isolated under `smoke/` and never count as completed full runs.

The notebook stops on a training error instead of retrying forever. Correct the
reported issue and rerun the training cell. Changing the training protocol should
use a new output directory rather than mixing different settings in one result.

## Run from the repository

```bash
python more_experiments/exp/exp15_stage234_tissue/scripts/train.py \
  --data-root datasets/wound-staging-masks-exp-3 \
  --output-dir more_experiments/exp/exp15_stage234_tissue/outputs
```

Optional arguments include `--variants rgb,rgb_tissue`, `--folds 0,1,2`,
`--workers 2`, `--batch 16`, and `--epochs 100`. Leave the defaults unchanged
for the planned experiment. `--smoke` runs a small two-epoch check and writes
to the `smoke/` subdirectory of the selected output directory.

## Results to bring back

The `results/` directory contains fold results (`results.csv`), model summaries
(`summary.csv`), paired fold differences (`paired_comparison.csv`), per-stage
metrics (`per_class_metrics.csv`), confusion matrices (`confusion_matrices.csv`),
pooled held-out metrics (`pooled_metrics.csv`), and per-image test predictions
(`test_predictions.csv`). Protocol and exact split
manifests are included so the run can be audited. Fold summaries and pooled
out-of-fold metrics should be interpreted according to their reported scope.

The notebook's final cell packs the report directory into a ZIP, saves a copy in
Drive under `exports/`, and starts a download. Checkpoints remain in Drive and
are excluded from the compact ZIP by default.

## Optional Stage-1 gate evaluation

The main three-class experiment can run without any existing staging checkpoint.
After both three-class models are trained, an optional notebook cell evaluates
the full cascade on the original four-class test folds:

1. The existing fold-matched four-class RGB model decides whether a sample is
   Stage 1. A predicted Stage-1 label is retained.
2. Every other sample goes to the newly trained three-class `rgb` or
   `rgb_tissue` model, which can only output Stage 2, 3, or 4.

Both cascades share the same Stage-1 gate. Routing uses predictions, never true
test labels, and gate errors are counted in the full-system results. This adds
an inference evaluation, not a newly trained binary gate. `RUN_CASCADE` is
`False` by default. To enable it, supply the Task 3 checkpoints named
`mit_b1__corn_fold0.pt`, `mit_b1__corn_fold1.pt`, and `mit_b1__corn_fold2.pt` in
the configured `GATE_CHECKPOINTS_DIR` (by default
`MyDrive/thesis/task_3_outputs/checkpoints`). These must correspond to the same
original folds used by this experiment.

The evaluator verifies the original Task 3 checkpoint hashes and restores the
historical image paths using exp-3's `metadata.csv`; retain that file in the data
archive. It also verifies that each specialist finished training under the saved
protocol and that its checkpoint matches the completed fold report.

Unlike the old frozen four-class WP2 cascade, the second branch here is actually
trained only on Stages 2–4 and cannot predict Stage 1.

## Maintaining the portable notebook

After editing the experiment or its shared dependencies, refresh the notebook's
embedded source bundle with `python scripts/bundle_notebook.py` from this folder.
The bundle includes only source/configuration files, requirements, tests, and the
historical gate protocol; it contains no clinical images or model weights.
