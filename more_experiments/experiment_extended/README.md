# Extended Experiments — Advisor Requests

Implementation notes and exact commands for the five advisor-requested work
packages described in the brief. See `summary.md` for findings and
`results.csv` for the consolidated metrics table.

## Status at a glance

| WP | What | Status |
|----|------|--------|
| 1  | Exp-4/Exp-5 repeated per single tissue channel | **Code complete, not yet trained** — needs a Colab GPU session |
| 2  | Tissue fusion restricted to Stages 2–4 | Done |
| 3  | Rule-based bone/tendon check for Stage 3 vs 4 | Done (negative result) |
| 4  | Stage 3–4 error inspection (panels + case table) | Done |
| 5  | Depth cue for Stage 3 vs 4 | Done (negative result) |

Only WP1 needs GPU training; WP2–WP5 are pure inference/analysis over already
-trained checkpoints and ran locally (CPU-only).

## Shared assets reused (not retrained unless noted)

- `datasets/wound-staging-masks-exp-3/` — the 8-channel (RGB + 5 tissue
  probability maps) `.npy` dataset exp-3/4/5 already generated; same leakage
  -safe tissue predictions, same verified PIID 3-fold split
  (`task_3/src/splits.py build_folds`).
- `task_3/results/task_3_outputs/checkpoints/mit_b1__corn_fold{0,1,2}.pt` — the
  RGB-only baseline, frozen, reused by WP1 (exp5_tissue_only variant) and WP2.
- `more_experiments/exp/exp4_late_fusion/checkpoints/mit_b1__corn__exp4_latefusion_concat_fold{0,1,2}.pt`
  — the existing 5-channel late-fusion checkpoint, reused by WP2.
- `datasets/wound-stage-depth-anything-v2-small/` — existing per-image relative
  depth maps (Depth Anything V2 Small), reused as-is by WP5.
- Tissue channel order throughout: `[granulation, slough, necrosis, tendon, bone]`.

## Work Package 1 — combined single-channel Exp-4 / Exp-5

**Corrected scope (see "Known issues" below):** the advisor's actual note is
*"Tek kanal exp-4 ve exp-5 denenecek"* — 5 kanal yerine tüm dokular tek kanalda
verilsin (all five tissue probability maps collapsed into **one** channel, instead
of concatenating them separately), **not** five separate per-tissue-type runs. The
combined channel is a **per-pixel maximum** over the 5 tissue probabilities
(`max(granulation, slough, necrosis, tendon, bone)` at every pixel) —
tissue-**type** identity is discarded, only "how much evidence of ANY non-healthy
tissue type is here" survives. This directly tests whether exp-4/exp-5's 5-channel
fusion needs per-type distinction or whether a single collapsed magnitude channel
is enough.

**Status: code complete and locally smoke-tested (CPU/MPS, 1 fold, 2 epochs, tiny
subset — both architectures pass end to end, including exp-5's RGB
-baseline-reproduction check), zero real training runs executed yet.** This is the
only GPU-bound piece, so it ships as two Colab notebooks:

- `wp1_single_tissue_channel/exp4_concat/colab_train.ipynb`
- `wp1_single_tissue_channel/exp5_tissue_only/colab_train.ipynb`

Both mirror the existing `more_experiments/exp/exp4_late_fusion/colab_train.ipynb`
/ `exp5_residual_fusion/colab_train.ipynb` pattern exactly (mount Drive → clone
repo → fetch `wound-staging-masks-exp-3.zip` from Drive → install
`task_3/requirements.txt` → smoke-test → redirect checkpoints/results to Drive →
train → inspect results), plus the exp5 notebook's extra step of fetching/training
the frozen RGB baseline checkpoints it depends on.

To run: open either notebook in Google Colab (GPU runtime), make sure
`wound-staging-masks-exp-3.zip` (and, for the exp5 notebook, optionally
`task_3_outputs/checkpoints/*.pt`) exist under `drive/MyDrive/thesis/`, and run
top to bottom. Each trains 3 folds (one combined-tissue-channel config), writing
`results/results.csv` with one row per fold.

Equivalent local command (if you have a CUDA/MPS GPU locally instead of Colab):

```bash
cd more_experiments/experiment_extended/wp1_single_tissue_channel/exp4_concat/scripts
python train.py --workers 4

cd ../../exp5_tissue_only/scripts
python train.py --workers 4
```

**Scope decision:** only exp-4's `concat` fusion mode is covered here, not
`gated`. The brief allows the gated variant to be "handled only if it is already
considered part of Experiment 4 in the repository" — `more_experiments/results/comparison.csv`
lists `E4-concat` and `E4-gated` as two separate recorded variants, so this WP
treats `concat` as the primary Exp-4 architecture (consistent with exp-4's own
notebook, which also only runs `gated` as a conditional follow-up). If the
combined-channel `concat` result justifies it, the same dataset-level channel
merge can be added to a `gated` variant later — not done here to keep this WP to
the "minimal modification" instruction.

**Known issue (corrected mid-implementation, flagged for the record):** an
earlier version of this WP misread "tek kanal" ("single channel") as *five
separate experiments, each isolating one tissue type alone* (granulation-only,
slough-only, ...) rather than *all five collapsed into one combined channel*. That
version's `--tissue-channel {name}` sweep and per-type Colab notebooks were
discarded entirely and replaced with the combined-max implementation described
above, once the advisor's actual note was quoted directly. No training had been
run under the old (incorrect) interpretation, so nothing was lost — only
unexecuted code and notebooks were replaced.

## Work Package 2 — tissue fusion restricted to Stages 2–4

Pure inference, no training. See `wp2_stage1_excluded_fusion/README.md` for the
full mechanism writeup (how the restriction is technically realized, without
touching CORN semantics) and results.

```bash
cd more_experiments/experiment_extended/wp2_stage1_excluded_fusion/scripts
python cascade_eval.py
```

## Work Package 3 — rule-based bone/tendon check for Stage 3 vs 4

Threshold `tau` fit on validation-only contested cases, evaluated on test.

```bash
cd more_experiments/experiment_extended/wp3_bone_tendon_rule/scripts
python fit_and_evaluate_rule.py
```

**Known issue (flagged, not silently fixed):** `wp3_bone_tendon_rule/results/rule_summary.csv`'s
column headers `before_true3_pred4_errors` / `before_true4_pred3_errors` are
swapped relative to what the underlying boolean expression in
`fit_and_evaluate_rule.py` (lines 199–202) actually computes — the column named
`true3_pred4` in fact holds the **true=4, predicted=3** count and vice versa.
Confirmed against (a) the identical computation in
`wp5_depth_stage34/scripts/extend_rule_with_depth.py`, which labels the same
expressions correctly, and (b) WP4's independently-tallied panel folders
(`error_panels/error_3to4/` has 65 images, `error_panels/error_4to3/` has 37).
The correct values — used throughout `results.csv` and `summary.md` here — are
**3→4 errors = 65, 4→3 errors = 37** (out of 530 contested test pairs). The
source CSV itself was left untouched since it's another in-progress session's
output; whoever owns that script should swap the two key names on lines 210–211.

## Work Package 4 — Stage 3–4 error inspection

Already run; outputs live at the top level per the brief's proposed layout:

- `stage34_error_cases.csv` — 530 contested cases (193 correct stage_3, 235
  correct stage_4, 65 true3→pred4, 37 true4→pred3), with wound-detection,
  bone/tendon mean+max, and confidence/entropy columns.
- `error_panels/{correct_3,correct_4,error_3to4,error_4to3}/` — review PNG panels.

```bash
cd more_experiments/experiment_extended/wp4_stage34_error_audit/scripts
python build_case_table.py
python render_panels.py
```

## Work Package 5 — depth cue for Stage 3 vs 4

Reuses the existing `datasets/wound-stage-depth-anything-v2-small/` outputs;
scoped only to the Stage 3/4 decision, not a full depth-fusion rerun.

```bash
cd more_experiments/experiment_extended/wp5_depth_stage34/scripts
python analyze_depth_signal.py
python extend_rule_with_depth.py
```

## Other notes / limitations

- Estimated depth (WP5) is treated throughout as a **relative visual cue**
  (Depth Anything V2 Small, monocular, unitless), never as measured physical
  wound depth.
- Tissue probability maps (WP3/WP4) are evidence of visibility in a 2-D
  projection, not proof of true exposed anatomy; absence of detected
  bone/tendon is not treated as proof of Stage 3 (Stage 4 can involve other
  deeper structures not captured by these two channels).
- WP3's and WP5's rules were fit on validation-only contested cases and then
  evaluated once on the same held-out test folds used everywhere else in this
  repo; no threshold was tuned against test labels. Both produced a **true
  negative result** (`n_changed_by_rule=0`) — see `summary.md` for the caveat
  this implies about what the validation-fit threshold could ever detect.
- No existing experiment results were overwritten; everything here lives under
  `more_experiments/experiment_extended/`.
- Nothing in this package has been committed to git — left as working-tree
  changes per instruction.
