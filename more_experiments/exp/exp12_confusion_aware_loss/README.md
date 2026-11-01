# Exp-12 — Confusion-Aware CORN Loss

**Status:** scaffolded, unit-tested (`scripts/test_losses.py`, 16/16 PASS),
local pipeline smoke-tested end to end (loss → cost matrices → train
select/final → evaluate → report). Frozen per-fold cost matrices already
computed from the existing baseline checkpoints
(`results/cost_matrices/fold{0,1,2}/`). **No real training has run yet —
the hyperparameter grid (Section 4 below) and the final 3-fold evaluation
both require a GPU and go through `colab_train.ipynb`.**

No new architecture, dataset, or encoder. The question is narrower and
sharper: does **how the CORN training objective is shaped** — by ordinal
distance alone, by empirical confusion between specific stages, or by
extra weight on just the hardest boundary — move accuracy at the stage
pairs the RGB baseline actually confuses, without hurting the rest?

## 1. Research question

> Can confusion-aware loss weighting improve ordinal pressure injury
> staging, particularly at frequently confused stage boundaries, without
> degrading overall classification performance?

Baseline (`task_3/results/task_3_outputs/results/ablation_summary.csv`,
`mit_b1__corn` row — `mit_b1 + CORN`, ImageNet weights, 3-fold):

| Metric | Baseline |
|---|---|
| Accuracy | 0.8168 ± 0.0150 |
| Macro-F1 | 0.8182 ± 0.0138 |
| MAE | 0.1896 ± 0.0204 |
| QWK | 0.9138 ± 0.0164 |

Held-out error breakdown (`more_experiments/reports/figures/mit_b1_corn_test_explanations/all_predictions.csv`,
1091 pooled test predictions): **200/1091 errors (18.3%)**, and **102 of
those 200 (51%)** are Stage 3↔4 confusions (65 Stage 3→4, 37 Stage 4→3) —
by far the single most concentrated error mode, confirmed independently
per-fold (see the chat transcript's fold-by-fold confusion matrices).

## 2. Hypothesis and the three variants it's split into

CORN already encodes an ordinal structure, but its training objective
treats every adjacent-stage mistake the same regardless of how often the
baseline actually makes it. If hard boundaries are weighted more in
training, that boundary's separation might improve.

- **Exp-12A — Ordinal Distance Regularization (control).** Does a general
  "stay close in ordinal distance" penalty help at all, with no confusion
  statistics?
- **Exp-12B — Confusion-Aware Ordinal Loss (the actual hypothesis).** Does
  knowing *which* stage pairs the baseline actually confuses, not just
  their ordinal distance, add anything beyond 12A?
- **Exp-12C — Boundary-Weighted CORN (simpler alternative).** Is a complete
  confusion-aware cost matrix even necessary, or does extra weight on just
  the Stage-3/4 CORN threshold already do the job?

The thesis-relevant distinction: if 12A improves but 12B doesn't, the
benefit (if any) comes from ordinal regularization in general, not from
confusion information specifically. 12B beating 12A is the stronger result
in favor of this experiment's actual hypothesis.

## 3. CORN probability reconstruction

For 4 ordered classes, CORN's 3 conditional logits give
`q_1=P(Y>1)`, `q_2=P(Y>2|Y>1)`, `q_3=P(Y>3|Y>2)`, reconstructed into class
probabilities `p_1=1-q_1`, `p_2=q_1(1-q_2)`, `p_3=q_1 q_2(1-q_3)`,
`p_4=q_1 q_2 q_3`. **Not reimplemented from scratch** — `scripts/losses.py`'s
`corn_class_probabilities()` is a thin wrapper around `task_3/corn_explain.py`'s
already-trusted `corn_outputs()` (used by the held-out explanation pipeline
that motivated this experiment), verified in `test_losses.py` to match this
exact closed form on a known `(q1,q2,q3)` triple, not just "sums to 1".
This reconstruction is used **only** for the three auxiliary losses below —
predicted labels still use `coral_pytorch`'s unmodified
`corn_label_from_logits` survival-threshold rule, unchanged.

## 4. Loss variants (`scripts/losses.py`)

All three are auxiliary terms **added** to the unchanged base `CornLoss`
(`task_3/src/losses.py`), selected via `CombinedLoss(variant=...)`.
`variant='baseline'` reproduces the original loss **exactly** — verified
`torch.equal` against `CornLoss` alone in `test_losses.py`, not just
numerically close.

**Exp-12A:** `L_A = L_CORN + lam * L_dist`,
`L_dist = mean_i sum_j p_{i,j} * |y_i-j| / 3`.

**Exp-12B:** `L_B = L_CORN + lam * L_conf`,
`L_conf = mean_i sum_j C[y_i,j] * p_{i,j}`, with the frozen cost matrix

```
C[a,b] = 0                                          if a == b
C[a,b] = (|a-b|/3) * (1 + alpha * E[a,b]/E_max)      if a != b
```

`E[a,b]` = row-normalized (true-class-conditioned) confusion rate from the
**train-side validation** predictions of the *already-trained baseline
checkpoint* (`task_3/results/task_3_outputs/checkpoints/mit_b1__corn_fold{i}.pt`),
never the outer test fold. Falls back to pure ordinal distance if every
off-diagonal validation confusion is zero (undefined `E_max`). Built once
per fold by `scripts/build_cost_matrices.py` (no training — a few seconds,
CPU-only) and frozen before any variant-B run; already computed and
committed under `results/cost_matrices/fold{0,1,2}/`:

| Fold | Val n | Val accuracy | Most confused pair |
|---|---|---|---|
| 0 | 73 | 0.9315 | Stage 3→4 (2) |
| 1 | 73 | 0.8356 | Stage 4→3 (2), Stage 3→4 (8) |
| 2 | 72 | 0.8333 | Stage 3→4 (5) |

(Validation accuracy is naturally higher than the ~0.82 test accuracy —
these are the same samples early stopping selected the checkpoint on.)

**Exp-12C:** `L_C = L_CORN + beta * L_3`, where `L_3` is the mean BCE for
CORN's **third** conditional threshold (the Stage-3/4 boundary) only,
restricted to its eligible Stage-3/4 samples — same eligibility rule
`coral_pytorch`'s own `corn_loss` uses internally for that task index,
reproduced here for exactly one threshold.

## 5. Training / selection protocol

Reuses `task_3/src`'s dataset/model/augmentations/splits/metrics/utils
**unchanged** — same `mit_b1 + CORN`, same `datasets/wound-stage-classification/`
images, same `build_folds(seed=42)` 3-fold split as the baseline.
`scripts/engine.py` is a **local copy** of `task_3/src/engine.py`'s
`fit()`/`run_epoch()`, changed in exactly one way: the criterion is passed
in by the caller instead of rebuilt internally from `(head, num_classes)` —
necessary because `task_3`'s own `fit()` can't be handed a `CombinedLoss`
with a selected variant/lambda/alpha/beta.

`scripts/train.py --mode select` trains on `{train}`, evaluates only on
`{val}`, and never touches `{test}` — used for the small grids below.
`--mode final` re-trains the same way, then evaluates **once** on the
outer `{test}` split, saves the checkpoint + every test-set prediction
(path, true/pred label, CORN logits, conditional and class probabilities).

| Variant | Grid |
|---|---|
| Exp-12A | lam ∈ {0.05, 0.10, 0.20} |
| Exp-12B | lam ∈ {0.05, 0.10, 0.20}, alpha ∈ {1, 2} |
| Exp-12C | beta ∈ {0.5, 1.0} |

Selection metric: validation macro-F1. Per the brief: **hyperparameters
are never selected using test results.** Grids run on fold 0 first
(`colab_train.ipynb` Section 6); every selected configuration is then
evaluated on all 3 folds (Section 7) for the reported comparison — a
single fold's validation ranking is not the thesis's result on its own.

## 6. Evaluation (`scripts/evaluate.py`, `scripts/generate_report.py`)

Per variant: pooled + per-fold 4×4 confusion matrices, per-stage
precision/recall/F1, every directed stage-pair error count, Stage 3→4 and
4→3 counts separately, adjacent-vs-non-adjacent error totals, and a
**paired fixed/regressed comparison against this same codebase's own
`variant='baseline'` reproduction**, matched by image path — not just
whether aggregate accuracy moved. A variant is **not** considered better
just because it reduces Stage 3→4 errors while increasing Stage 4→3 errors
or hurting overall accuracy (brief, Section 6).

## 7. Pipeline

```bash
cd scripts
python test_losses.py              # run first -- must be ALL CHECKS PASSED
python build_cost_matrices.py       # already run once; re-run only if the baseline checkpoints change

# selection (fold 0, validation only)
python train.py --variant A --fold 0 --mode select --lam 0.10
python train.py --variant B --fold 0 --mode select --lam 0.10 --alpha 1
python train.py --variant C --fold 0 --mode select --beta 0.5

# final (all 3 folds, after hyperparameters are chosen from the selection table)
python train.py --variant baseline --fold 0 --mode final
python train.py --variant B --fold 0 --mode final --lam 0.10 --alpha 1
# ... repeat per variant per fold ...

python evaluate.py --config-id mit_b1__corn__exp12B_lam0.1_alpha1
python generate_report.py --config-ids mit_b1__corn__exp12_baseline mit_b1__corn__exp12B_lam0.1_alpha1 ...
```

`colab_train.ipynb` runs this entire sequence on a GPU runtime, including
the full Section 4/5 grids and the final 3-fold evaluation.

## 8. Experimental integrity

The existing held-out baseline predictions (and this chapter's own
Stage-3/4-concentration finding) motivated this hypothesis — same-fold
comparisons here are **exploratory**, not an untouched independent
confirmation of a hypothesis formed before looking at any data. No claim
of statistical significance or novelty is made solely from an improvement
in this experiment. This experiment does not modify any other experiment's
results or checkpoints.
