# Extended Experiments — Findings Summary

Status overview and exact commands are in `README.md`; full per-metric numbers are
in `results.csv`. This file answers the five advisor questions directly, flags
what's exploratory vs. confirmatory, and lists limitations.

## 1. Does any individual tissue channel carry signal the 5-channel fusion hides?

**Not yet answered — pending Colab training (WP1).** The code for both
architectures (Exp-4 concat and Exp-5 `tissue_only` residual, swept over
granulation/slough/necrosis/tendon/bone × 3 folds) is complete and was validated to
build, load, and run via `--smoke` mode, but no GPU training has been executed in
this environment. Two Colab notebooks
(`wp1_single_tissue_channel/{exp4_concat,exp5_tissue_only}/colab_train.ipynb`) are
ready to run; once they finish, `results.csv`'s `wp1` rows should be filled in with
real per-channel accuracy/macro-F1/MAE/QWK and compared against the recorded
references already in that file (RGB baseline 0.8168/0.8182/0.9138; Exp-4 concat
5-channel 0.7983/0.8003/0.9109; Exp-5 `tissue_only` 5-channel, fold-1-only,
0.8370/0.8376/0.9323).

## 2. Does RGB+tissue fusion help Stages 2–4 if Stage 1 is excluded from fusion?

**No — negative result (WP2).** A frozen-model inference cascade (RGB baseline
decides; its own `stage_1` predictions pass through untouched; everything else
defers to the already-trained 5-channel Exp-4 fusion checkpoint) does not recover
what full fusion loses. Restricted to Stages 2–4 specifically — the scope this
question targets — the RGB baseline alone (0.8002 accuracy) beats both the
unrestricted fusion model (0.7724) and the Stage-1-excluded cascade (0.7666). The
cascade is mathematically guaranteed to leave Stage-1 decisions unchanged (verified
exactly: 223/223 baseline-`stage_1` predictions preserved, and the remaining 868
predictions match the unrestricted fusion model exactly) — so the restriction
itself introduces no instability, it simply inherits the same underlying weakness
of Exp-4's fusion checkpoint on Stages 2–4. Full details and the exact mechanism
are in `wp2_stage1_excluded_fusion/README.md`.

## 3. Does a bone/tendon rule correct Stage 3 vs 4 mistakes?

**No — negative result (WP3).** A single threshold `tau` on `max(bone_evidence,
tendon_evidence)` was fit on 110 validation-only contested cases and evaluated once
on 530 held-out test contested cases (same split used everywhere else). The fit
threshold (`tau=1.0`) never fires on the test set: `n_changed_by_rule=0`, so
before/after pair-accuracy is identical (0.8075) and both error directions are
unchanged: **3→4 errors stay at 65, 4→3 errors stay at 37** (these two counts are
corrected from the source CSV — see Known Issues below). Zero corrections, zero
damage — the rule as fit is simply inert on this data, not a rule that trades gains
for losses.

This is consistent with what WP4's case table shows numerically: bone/tendon
probability-map values are tiny across *all* four groups (correct and error alike
— max values in the 0.04–0.065 range even for correctly classified Stage 4 cases),
and critically, the **error_4to3** group (true Stage 4, misclassified as Stage 3)
has a *higher* mean `bone_max` (0.0652) than the **correct_4** group (0.0546) — so
"more bone evidence" does not cleanly separate correct Stage-4 calls from missed
ones in this sample, which is exactly the kind of signal a single global threshold
can't exploit. A finer-grained rule (e.g. per-pixel spatial checks, or a
rule that also accounts for visibility/occlusion) might behave differently, but
that's outside what WP3 was asked to do (stay rule-based, no new learned network).

## 4. Are there consistent clues in Stage 3–4 mistakes? (exploratory)

Materials are ready for a human visual review:
`stage34_error_cases.csv` (530 rows) and `error_panels/{correct_3,correct_4,error_3to4,error_4to3}/`
(20/20/65/37 PNG panels — every error case has a panel; correct cases are a
20-image illustrative sample of 193/235 total). Each panel shows RGB, bone-map
overlay, and tendon-map overlay side by side, labeled with true/predicted stage and
P(pred)/entropy.

**Measured patterns (from the full 530-row case table, not visual guesses):**
- Both error groups are lower-confidence than their correct-group counterparts:
  mean `P(pred)` is 0.8405 (correct_3) vs. 0.8150 (error_3to4), and 0.8858
  (correct_4) vs. **0.7541** (error_4to3) — the model's own confidence is
  measurably lower on the cases it gets wrong, most sharply for 4→3 misses.
- Entropy mirrors this: 0.38 (correct_3) vs. 0.42 (error_3to4); 0.29 (correct_4)
  vs. **0.48** (error_4to3).
- Bone/tendon mean/max values are small and do not cleanly separate groups (see
  §3) — in particular, `error_4to3` has *higher* bone signal on average than
  `correct_4`, contradicting a naive "more bone detected → more likely Stage 4" story.
- `wound_detected` is `True` for all 530 cases in all four groups — none of these
  errors trace back to a wound-detection failure.
- `wound_area_fraction` is somewhat larger in error groups than their correct
  counterparts (0.338 → 0.409 for the 3-side; 0.396 → 0.414 for the 4-side) but the
  gap is small and not verified for statistical significance here.

**Illustrative single-example observations (NOT a verified pattern — two panels
inspected, not a systematic visual review):**
- One high-confidence 3→4 error (`stage_3_10062`, P(pred)=0.971, bone/tendon ≈ 0)
  shows a wound bed with dark, shadowed, visually "deep"-looking cavitated tissue
  despite ground truth being Stage 3 — plausibly consistent with the brief's
  suggestion that perceived cavity/depth might drive some errors, but this is one
  image, not a confirmed trend.
- One 4→3 miss (`stage_4_12553`, P(pred)=0.703 for the *wrong* label, bone/tendon ≈
  0) shows a wound largely covered by a pale slough patch obscuring most of the
  bed — consistent with the brief's caution that absence of detected bone/tendon
  does not prove Stage 3, since the true deeper structures may simply not be
  visible/segmented under overlying slough.

A full visual pass over all 102 error panels (and a comparison sample of the
correct ones) by a human reviewer is the natural next step to turn these two
illustrative examples and the four measured patterns above into a confirmed list
of recurring clues; that pass was not performed exhaustively in this session.

## 5. Does depth help distinguish Stage 3 from Stage 4?

**Mixed signal, but adds nothing once combined with the WP3 rule — negative result
(WP5).** Treating Depth Anything V2 Small's output strictly as a *relative visual
cue* (not measured physical depth), four depth-derived statistics were compared
between correct and incorrect predictions within each true stage (530 cases,
t-tests, exploratory — not a train/test-separated claim):

| true stage | statistic | p-value | correct vs. wrong mean |
|---|---|---|---|
| stage_3 | `wound_depth_range` | 0.0193 | 1.660 vs. 1.906 (wrong cases show *more* range) |
| stage_3 | `wound_minus_skin_mean` | 0.0327 | 0.131 vs. −0.086 |
| stage_4 | `wound_depth_std` | 0.0006 | 0.520 vs. 0.430 |
| stage_4 | `wound_depth_range` | 0.0042 | 2.163 vs. 1.842 |

Two of the four comparisons are significant at p<0.01 even with this modest sample,
so depth does carry *some* separable signal between correctly- and
incorrectly-classified cases. However, when a depth threshold
(`tau_depth=-1.54`, fit the same way as WP3's rule, on the same 110 validation
contested cases) was added on top of WP3's bone/tendon rule and evaluated on the
same 530 test pairs, it **changed nothing**: `n_changed_by_rule=0`,
before/after pair-accuracy identical (0.8075), both error directions unchanged.
The validation-fit threshold never crosses into the regime that would flip a test
prediction — the same "inert rule" outcome as WP3, for the same underlying reason
(a single global threshold fit on 110 cases doesn't reliably transfer even when
group-level means do differ).

## Known issues / things to double-check

- **WP3 CSV column-label bug (confirmed, not fixed in place):**
  `wp3_bone_tendon_rule/results/rule_summary.csv`'s `before_true3_pred4_errors` /
  `before_true4_pred3_errors` columns are swapped relative to what
  `fit_and_evaluate_rule.py` (lines 199–202, 210–211) actually computes. Verified
  against the equivalent, correctly-labeled computation in
  `wp5_depth_stage34/scripts/extend_rule_with_depth.py` and against WP4's
  independently-tallied panel folder counts (`error_3to4/`=65 images,
  `error_4to3/`=37 images). The **correct** values, used throughout this file and
  `results.csv`, are: 3→4 errors = 65, 4→3 errors = 37 (both unchanged by the
  rule). Left unfixed in the source CSV since it's another in-progress session's
  script — flagged here and in `README.md` for whoever owns
  `wp3_bone_tendon_rule/scripts/fit_and_evaluate_rule.py` to correct (swap the two
  key names on lines 210–211).
- WP1 is the only unfinished piece — everything in this summary that depends on
  single-channel results is explicitly marked pending above, not fabricated.
- §4's illustrative single-example observations are clearly separated from the
  measured, full-sample patterns — treat the former as hypotheses for a human
  reviewer to check against the full panel set, not as findings.

## Limitations (apply across all work packages)

- Tissue probability maps (bone, tendon, and the other three channels) are
  evidence of *visibility* in a 2-D segmentation, not proof of true exposed
  anatomy — absence of detected bone/tendon is never treated as proof of Stage 3;
  Stage 4 can involve deeper structures these two channels don't capture.
- Estimated depth (WP5) is a monocular, unitless, *relative* visual cue — never
  treated as measured physical wound depth.
- WP3/WP5 rule-fitting used validation-only contested cases and a single held-out
  test evaluation — no threshold was tuned against test labels, but the WP4 error
  audit that *informed* where to look (bone/tendon, depth) was itself run on the
  same three test folds used for final evaluation here, so by the brief's own
  safeguard, these are exploratory confirmations on those folds, not a fully
  independent test of the hypothesis-generation process.
- No existing experiment results were overwritten or retrained; all comparisons
  against the RGB baseline, Exp-3, Exp-4, and Exp-5 reuse already-recorded numbers
  from `more_experiments/results/comparison.csv` and each experiment's own
  `results/` directory.
