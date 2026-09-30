# Dual-Head Multi-Task Proposal (Stage Classification + Segmentation)

Advisor suggestion (2026-09-30): instead of (or in addition to) the full
task_1 architecture ablation, explore a dual-head network that shares an
encoder between wound staging and a segmentation task, as a distinct
thesis contribution beyond replicating the paper's benchmark.

## Motivation

The paper (`segmentation_benchmarking.pdf`) trains wound region
segmentation and intra-wound tissue segmentation as two independent,
sequentially-chained models (Stage 1 crops the wound, Stage 2 classifies
tissue within the crop). Wound stage classification (Stage 1-4 severity)
isn't part of that pipeline at all — it's a separate dataset, currently
unused in task_1.

A shared-encoder multi-task model tests whether joint training transfers
useful representations across tasks, which matters most for the smallest
dataset here (stage classification: 1,091 images) that's otherwise prone
to overfitting on its own.

## Dataset field-of-view finding

Inspecting sample images from all three datasets (`datasets/*/`) before
committing to a pairing:

- **Region dataset**: wide shot, full body part + background visible.
- **Tissue dataset**: tight close-up, wound interior only, no skin/background
  context.
- **Stage dataset**: spans both ends depending on severity — Stage 1 samples
  look like the region dataset (intact skin, discoloration, no open wound);
  Stage 3-4 samples look like the tissue dataset (deep open wound, wound bed
  dominates the frame, minimal skin context).

Conclusion: stage classification isn't a single visual domain, so no single
pairing (region+stage or tissue+stage) covers the whole severity spectrum
well on its own. Rather than force a single 3-way shared encoder (harder to
balance, more failure modes to debug in a week), train **two separate
dual-head models** and compare which pairing helps staging more, and where.

## Proposed models

### Model 1: Region + Stage
Shared encoder, two heads:
- Segmentation head → wound region mask (background/skin/wound), reusing
  task_1's `wound-region-segmentation` dataset.
- Classification head → wound stage (1-4), using `wound-stage-classification`.

Hypothesis: region-segmentation features (skin condition, wound boundary)
help most on early stages, where the visual signal is subtle skin change
rather than an open wound.

### Model 2: Tissue + Stage
Shared encoder, two heads:
- Segmentation head → tissue mask (granulation/slough/necrosis), using
  `wound-tissue-segmentation`.
- Classification head → wound stage (1-4), using `wound-stage-classification`.

Hypothesis: tissue features (wound bed depth/composition) help most on
later stages, where the segmentation and staging tasks are visually most
similar.

### Baseline
A stage-only classifier (no shared encoder, no auxiliary segmentation loss)
trained on the same data, same encoder/backbone, same protocol — the
control both dual-head models are compared against. Without this, "did
joint training help" isn't measurable.

## Training data pipeline

The two datasets in each model are **not paired** — different photos,
different patients, no overlap. Joint training happens at the *optimizer
step* level, not the *image* level:

```
each training step:
    batch_A = next(segmentation_loader)      # image + mask
    batch_B = next(stage_loader)             # image + stage label

    feats_A = encoder(batch_A.image)
    feats_B = encoder(batch_B.image)

    loss_seg = seg_loss(seg_head(feats_A), batch_A.mask)
    loss_cls = cls_loss(cls_head(feats_B), batch_B.label)

    loss = loss_seg + lambda_cls * loss_cls
    loss.backward()
    optimizer.step()
```

Practical issues to handle:

- **Dataset size mismatch** (region 3,476 / tissue 568 / stage 1,091): wrap
  the smaller loader in `itertools.cycle` so it never runs dry; let the
  larger dataset define one epoch's length.
- **Loss scale mismatch**: segmentation loss (CE+Dice, as in
  `task_1/src/losses.py`) and classification loss (cross-entropy over 4
  classes) aren't naturally on the same scale — `lambda_cls` needs tuning
  empirically (start at 1.0, adjust based on validation staging accuracy
  vs. segmentation mIoU both moving sensibly).

## Evaluation plan

- Stage classification: accuracy, macro-F1, confusion matrix (4 classes,
  likely imbalanced per the dataset README counts: 230/313/275/273).
- Segmentation head: same mIoU/mDice protocol as task_1, mostly as a sanity
  check that the auxiliary task is actually learning something, not the
  primary metric of interest.
- Primary comparison: baseline stage-only accuracy vs. Model 1 vs. Model 2,
  broken down **per stage** to check whether the region/tissue pairing
  helps the stage range it was hypothesized to help (early vs. late).

## Open questions (resolve during implementation)

- Which architecture/encoder to use for the shared trunk — reuse task_1's
  best performer (U-Net + MiT-B2) as a starting point, or pick something
  lighter given the smaller datasets?
- Split protocol for the two new datasets (k-fold like task_1, or a single
  held-out split given the 1-week timeline)?
- Exact value/schedule for `lambda_cls` — fixed, or annealed during
  training?
- How to structure the code: new `task_2/` (tissue+stage) and `task_3/`
  (region+stage) dirs, or one shared multi-task package since the training
  loop structure is identical between the two models?

## Timeline

Target: implemented and running within 1 week of 2026-09-30.
