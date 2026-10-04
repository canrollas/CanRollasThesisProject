# Field-of-View Ablation for Wound Stage Classification

## Status (2026-10-02)

The dual-head shared-encoder idea below (Model 1 / Model 2 / baseline, joint
training) is **dropped**. It doesn't fit a 1-month timeline on top of the
rest of the thesis: Task 3 (tissue) and Task 2 (stage) don't have their own
baseline models trained yet, the shared-trunk design still had open
questions (architecture, `lambda_cls`, split protocol), and joint training
is inherently iterative — no guarantee the first attempt shows anything.

What's kept from it is the one finding that's actually cheap to act on: the
**dataset field-of-view finding** below. It motivates a much lighter
ablation that needs no joint training and no new datasets — just the
already-trained Task 1 region model and three ordinary stage classifiers.

## Motivation

The paper (`segmentation_benchmarking.pdf`) trains wound region
segmentation and intra-wound tissue segmentation as two independent,
sequentially-chained models (Task 1 crops the wound, then the tissue model
classifies tissue within that crop). Wound stage classification (Stage 1-4
severity) isn't part of that chain at all: PIID has no spatial annotation,
so it is currently classified from the raw, uncropped photograph.

## Dataset field-of-view finding

Inspecting sample images from all three datasets (`datasets/*/`):

- **Region dataset**: wide shot, full body part + background visible.
- **Tissue dataset**: tight close-up, wound interior only, no skin/background
  context.
- **Stage dataset**: spans both ends depending on severity — Stage 1 samples
  look like the region dataset (intact skin, discoloration, no open wound);
  Stage 3-4 samples look like the tissue dataset (deep open wound, wound bed
  dominates the frame, minimal skin context).

This is the open question worth testing on its own: if a stage classifier's
accuracy depends on how tightly the input is cropped around the wound, then
the pipeline's choice to classify stage from the *raw, uncropped* photograph
is itself a design decision that hasn't been tested against the alternative
of classifying from a crop produced by Task 1.

## Proposed ablation

Train the **same classifier architecture/backbone, same training protocol**,
three times, on three different crops of the stage dataset. No shared
encoder, no auxiliary segmentation loss, no joint training — just three
independent stage classifiers that differ only in what field of view they
see.

| Variant | Input | How it's produced |
|---|---|---|
| **Raw** | Full, uncropped PIID photograph | Already exists — current pipeline default |
| **Wide crop** | Bounding box around wound + surrounding skin, with margin | Run the trained Task 1 model (best config: MAnet + MiT-B2, see `task_1/results/.../ablation_summary.csv`) on each PIID image, take the union of predicted wound+skin pixels, crop to that box |
| **Tight crop** | Bounding box around predicted wound pixels only, no margin | Same Task 1 inference pass, crop to the wound-only box |

Both crop variants reuse Task 1's **already-trained** checkpoint purely for
inference on PIID images — no new segmentation model, no new training run,
no dependency on Task 2 (which doesn't exist yet). This is the only new
engineering needed: a short inference script that runs the saved Task 1
checkpoint over `datasets/wound-stage-classification/` and writes out two
additional cropped copies of the dataset.

Hypothesis, carried over from the original field-of-view finding: the wide
crop should help early stages (1-2), where the signal is subtle skin change
best read with context; the tight crop should help, or at least not hurt,
late stages (3-4), where the wound bed already fills most of the raw frame
anyway; and if neither crop beats raw, that itself is a usable negative
result — it says stage classification doesn't benefit from localization,
which is worth knowing before anyone proposes chaining it onto Task 1 in a
future pipeline revision.

## Evaluation plan

- Accuracy, macro-F1, and confusion matrix per variant (4 classes, imbalance
  per the dataset README: 230/313/275/273).
- Primary comparison: raw vs. wide-crop vs. tight-crop accuracy, broken down
  **per stage**, to check whether cropping helps the stage range it's
  hypothesized to help (early vs. late) rather than just reading off one
  overall accuracy number.
- Same train/eval protocol (split, seed, epochs, optimiser) across all three
  variants, so any difference is attributable to the input crop and nothing
  else.

## Open questions (resolve during implementation)

- Margin for the wide crop (fixed pixel/percentage padding around the
  wound+skin box) — needs a value, not just "some margin".
- Split protocol for `wound-stage-classification` (k-fold like task_1, or a
  single held-out split).
- Whether to report per-variant results from a single run or average over
  folds/seeds, given this is now a much smaller experiment than the dropped
  dual-head plan and there's time budget to do it properly.

## Timeline

Much smaller scope than the dropped proposal: one inference pass with an
existing checkpoint, three ordinary classifier training runs. Target:
results in hand well inside the 1-month window.
