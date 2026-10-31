#!/usr/bin/env python3
"""Work Package 2 (advisor extended experiments): tissue fusion restricted
to Stages 2-4.

Mechanism (documented here before the code, per the brief):

  A two-stage CASCADE at inference, not a new architecture and not a
  ground-truth gate:

    1. Run the frozen RGB-only baseline (task_3's own trained
       mit_b1__corn checkpoint for this fold), decode with the SAME rule
       used everywhere else in this repo (corn_outputs()'s survival > 0.5
       median rule, task_3/corn_explain.py).
    2. If the baseline's OWN decoded label is stage_1, that IS the final
       prediction -- the tissue-fusion branch never runs on this sample.
    3. Otherwise, run exp-4's existing 5-channel late-fusion checkpoint
       (fusion="concat", the complete-3-fold variant) for this fold and
       use ITS decoded label as the final prediction.

  This never looks at the true label to pick a branch (the routing
  decision is the RGB baseline's own prediction, nothing else), and both
  branches decode with the identical, unmodified CORN rule -- no CORN
  semantics are changed, this is a selection between two already-trained
  models' outputs, nothing more.

No training happens here -- both checkpoints already exist
(task_3/results/task_3_outputs/checkpoints/mit_b1__corn_fold{fold}.pt and
more_experiments/exp/exp4_late_fusion/checkpoints/mit_b1__corn__exp4_latefusion_concat_fold{fold}.pt).
This script only runs frozen forward passes, fold by fold, over each
fold's own TEST split (task_3/src/splits.py's build_folds, called on
exp-4's 8-channel dataset root so the exact same images are available to
both branches -- same PIID/filename split as every other experiment
here).
"""
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent              # experiment_extended/wp2_stage1_excluded_fusion/scripts/
WP_DIR = THIS_DIR.parent
EXT_DIR = WP_DIR.parent
REPO_ROOT = THIS_DIR.parents[3]

sys.path.insert(0, str(REPO_ROOT / "task_3"))
from src.dataset import CLASS_ORDER  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.model import build_model as build_rgb_model  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.utils import get_device, load_yaml  # noqa: E402
from corn_explain import corn_outputs  # noqa: E402

EXP4_SCRIPTS = REPO_ROOT / "more_experiments/exp/exp4_late_fusion/scripts"
sys.path.insert(0, str(EXP4_SCRIPTS))
from dataset import TwoStreamStageDataset  # noqa: E402
from model import build_model as build_fusion_model  # noqa: E402

EXP4_CFG = load_yaml(REPO_ROOT / "more_experiments/exp/exp4_late_fusion/configs/base.yaml")
RGB_CKPT_TEMPLATE = REPO_ROOT / "task_3/results/task_3_outputs/checkpoints/mit_b1__corn_fold{fold}.pt"
FUSION_CKPT_TEMPLATE = (
    REPO_ROOT / "more_experiments/exp/exp4_late_fusion/checkpoints/mit_b1__corn__exp4_latefusion_concat_fold{fold}.pt"
)
DATA_ROOT = REPO_ROOT / EXP4_CFG["data_root"].replace("../../../", "")  # exp-4's own relative path, resolved from REPO_ROOT
STAGE1_IDX = 0
STAGES_2_TO_4 = {1, 2, 3}  # 0-indexed: stage_2, stage_3, stage_4

OUT_PER_SAMPLE = WP_DIR / "results" / "cascade_per_sample.csv"
OUT_METRICS = WP_DIR / "results" / "cascade_metrics.csv"


@torch.no_grad()
def decode(model_logits_fn, *inputs):
    logits = model_logits_fn(*inputs)
    out = corn_outputs(logits)
    return out["decoded"]  # [B] long tensor


def main():
    device = get_device()  # cuda > mps > cpu -- this repo's own convention (task_3/src/utils.py)
    num_classes = len(CLASS_ORDER)

    all_rows = []
    for fold_idx in range(3):
        rgb_ckpt = Path(str(RGB_CKPT_TEMPLATE).format(fold=fold_idx))
        fusion_ckpt = Path(str(FUSION_CKPT_TEMPLATE).format(fold=fold_idx))
        if not rgb_ckpt.is_file():
            raise SystemExit(f"BLOCKED: {rgb_ckpt} not found.")
        if not fusion_ckpt.is_file():
            raise SystemExit(f"BLOCKED: {fusion_ckpt} not found.")

        rgb_model = build_rgb_model("mit_b1", None, "corn", num_classes=num_classes)
        rgb_model.load_state_dict(torch.load(rgb_ckpt, map_location=device))
        rgb_model.to(device).eval()

        fusion_model = build_fusion_model("mit_b1", None, "concat", num_classes=num_classes)
        fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        fusion_model.to(device).eval()

        folds = build_folds(DATA_ROOT, EXP4_CFG["num_folds"], EXP4_CFG["val_fraction"], EXP4_CFG["seed"])
        test_samples = folds[fold_idx]["test"]
        ds = TwoStreamStageDataset(test_samples, DATA_ROOT, EXP4_CFG["image_size"], transform=None)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

        print(f"[fold{fold_idx}] evaluating {len(test_samples)} test images "
              f"(rgb={rgb_ckpt.name}, fusion={fusion_ckpt.name})", flush=True)

        sample_idx = 0
        for rgb, tissue, targets in loader:
            rgb, tissue = rgb.to(device), tissue.to(device)
            baseline_decoded = decode(lambda x: rgb_model(x), rgb)
            fusion_decoded = decode(lambda r, t: fusion_model(r, t), rgb, tissue)
            cascade_decoded = torch.where(baseline_decoded == STAGE1_IDX, baseline_decoded, fusion_decoded)

            for i in range(len(targets)):
                rel_path, true_idx = test_samples[sample_idx]
                all_rows.append({
                    "path": rel_path, "fold": fold_idx,
                    "true": int(true_idx),
                    "baseline_pred": int(baseline_decoded[i]),
                    "fusion_pred": int(fusion_decoded[i]),
                    "cascade_pred": int(cascade_decoded[i]),
                })
                sample_idx += 1

        del rgb_model, fusion_model

    OUT_PER_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PER_SAMPLE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {OUT_PER_SAMPLE} ({len(all_rows)} rows)")

    # Metrics: per model (baseline / fusion-only / cascade), per scope (all
    # stages / stages 2-4 only, i.e. true in {stage_2,stage_3,stage_4}),
    # overall (pooled across all 3 folds' own test sets, standard practice
    # for this repo's 3-fold protocol -- see task_3/run_ablation.py).
    metric_rows = []
    for model_name in ("baseline", "fusion", "cascade"):
        pred_key = f"{model_name}_pred" if model_name != "baseline" else "baseline_pred"
        for scope, scope_name in ((None, "all_stages"), (STAGES_2_TO_4, "stages_2_4")):
            y_true, y_pred = [], []
            for row in all_rows:
                if scope is not None and row["true"] not in scope:
                    continue
                y_true.append(row["true"])
                y_pred.append(row[pred_key])
            stats = compute_metrics(y_true, y_pred, num_classes)
            metric_rows.append({
                "model": model_name, "scope": scope_name, "n": len(y_true),
                "accuracy": round(stats["accuracy"], 4), "macro_f1": round(stats["macro_f1"], 4),
                "mae": round(stats["mae"], 4), "qwk": round(stats["qwk"], 4),
            })

    # Stage-1 stability check: among samples the RGB baseline itself decoded
    # as stage_1, cascade_pred must equal baseline_pred by construction --
    # verified explicitly rather than just assumed.
    stage1_routed = [r for r in all_rows if r["baseline_pred"] == STAGE1_IDX]
    stage1_stable = all(r["cascade_pred"] == r["baseline_pred"] for r in stage1_routed)
    # Secondary check: true stage_1 samples the baseline itself got wrong
    # (predicted non-stage_1) get routed into the fusion branch instead --
    # how many, and does the fusion branch's answer help or hurt there.
    true_stage1_missed_by_baseline = [
        r for r in all_rows if r["true"] == STAGE1_IDX and r["baseline_pred"] != STAGE1_IDX
    ]
    rescued = sum(1 for r in true_stage1_missed_by_baseline if r["cascade_pred"] == STAGE1_IDX)

    with open(OUT_METRICS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"wrote {OUT_METRICS}")
    for row in metric_rows:
        print(f"  {row}")
    print(f"\nstage-1 routing stability: {len(stage1_routed)} samples routed to baseline-only "
          f"(baseline predicted stage_1), all cascade-consistent={stage1_stable}")
    print(f"true-stage_1 samples the baseline itself misclassified as non-stage_1: "
          f"{len(true_stage1_missed_by_baseline)} (routed into the fusion branch instead); "
          f"fusion branch then got {rescued} of those back to stage_1 correctly")


if __name__ == "__main__":
    main()
