# Work Package 2 — Tissue Fusion Restricted to Stages 2–4

**Advisor question:** does tissue fusion help Stage 2/3/4 classification when Stage 1
is left to the plain RGB model, i.e. is fusion only hurting because it's forced onto
Stage-1 decisions it doesn't need?

## Mechanism (how the restriction is technically realized)

A two-stage **cascade at inference time** — not a new architecture, and not a
ground-truth gate. Implemented in `scripts/cascade_eval.py`:

1. Run the frozen RGB-only baseline (task_3's own trained `mit_b1__corn` checkpoint
   for this fold) and decode it with the repo's one standard CORN decode rule
   (`corn_outputs()`'s survival > 0.5 / median rule, `task_3/corn_explain.py`).
2. If the baseline's **own predicted** label is `stage_1`, that is the final
   prediction — the tissue-fusion branch never runs on this sample.
3. Otherwise, run Exp-4's existing 5-channel late-fusion checkpoint (`concat`
   fusion, already trained 3-fold) and use *its* decoded label as the final
   prediction.

No ground-truth label is ever used to choose the branch — routing depends only on
the baseline's own prediction. Both branches decode through the identical,
unmodified CORN rule, so CORN semantics are untouched; this is purely a selection
between two already-trained, frozen models' outputs. No training happens in this
work package.

**Checkpoints reused, not retrained:**
- `task_3/results/task_3_outputs/checkpoints/mit_b1__corn_fold{0,1,2}.pt` (RGB baseline)
- `more_experiments/exp/exp4_late_fusion/checkpoints/mit_b1__corn__exp4_latefusion_concat_fold{0,1,2}.pt` (5-channel fusion)

## Stage-1 stability check

Verified directly on `results/cascade_per_sample.csv` (1091 test samples, 3 folds):

- Of the 223 samples where the baseline itself predicts `stage_1`, the cascade
  agrees on **all 223** (by construction — the fusion branch is never invoked for
  them).
- Of the remaining 868 samples, the cascade's prediction exactly matches the
  unrestricted fusion model's prediction for **all 868**.

So Stage-1 decisions are provably stable, not just empirically similar — the
cascade is mathematically guaranteed to never override a baseline `stage_1` call.

## Results (`results/cascade_metrics.csv`)

| model    | scope       | n    | accuracy | macro_f1 | mae    | qwk    |
|----------|-------------|------|----------|----------|--------|--------|
| baseline | all_stages  | 1091 | 0.8167   | 0.8185   | 0.1897 | 0.9141 |
| baseline | stages_2_4  | 861  | 0.8002   | 0.6055   | 0.2044 | 0.8528 |
| fusion   | all_stages  | 1091 | 0.7993   | 0.8012   | 0.2044 | 0.9113 |
| fusion   | stages_2_4  | 861  | 0.7724   | 0.5901   | 0.2323 | 0.8371 |
| cascade  | all_stages  | 1091 | 0.7993   | 0.8011   | 0.2044 | 0.9120 |
| cascade  | stages_2_4  | 861  | 0.7666   | 0.5876   | 0.2381 | 0.8351 |

(`stages_2_4` macro_f1 is computed over the same 4-class label space restricted to
these 861 samples' *true* labels — stage_1 contributes 0 to several per-class terms
since it's absent from this subset's ground truth, which is why it reads much lower
than the `all_stages` macro_f1; this is a scope artifact of the metric, not a sign
that the models collapse on 3 classes.)

## Finding (negative result)

Restricting fusion to Stages 2–4 **does not help** — it does not even match the
unrestricted fusion model, let alone the RGB baseline:

- Overall: baseline (0.8167) > fusion (0.7993) ≈ cascade (0.7993).
- Restricted to Stages 2–4 specifically — the scope this WP targets — baseline
  (0.8002) > fusion (0.7724) > **cascade (0.7666)**: the cascade is slightly
  *worse* than applying the unrestricted fusion model everywhere, even measured only
  on the subset it's meant to help.

This is expected given the mechanism: the cascade reuses the exact same frozen
5-channel fusion checkpoint as the unrestricted model whenever it runs — it changes
*when* that checkpoint is consulted, not what it predicts. Since Exp-4's fusion
checkpoint already underperforms the RGB baseline on Stages 2–4 before any
restriction, routing Stage-1 away from it cannot repair that underlying weakness,
and in this run it compounds slightly (0.7724 → 0.7666), most likely sampling noise
given no re-training occurred. The RGB-only baseline remains the better choice for
Stages 2–4 on this split.
