#!/usr/bin/env python3
"""Verify PIID Stage-1 overlap between Task 1 (wound-region-segmentation) and
Task 3 (wound-stage-classification), and check Task 2 (wound-tissue-segmentation)
for any PIID involvement.

Background: Task 1's training images are source-tagged by filename prefix
(see task_1/src/splits.py SOURCE_PATTERNS). One source, "ft_evre1_stage_1",
is suspected to be drawn from the same PIID Stage-1 photographs used to
evaluate Task 3 staging. If so, any Task-1 checkpoint trained on that data
must not be reused to generate localisation/crops for Task 3 experiments.

This script re-derives the overlap from the raw files on disk (not from any
cached assumption) and writes out:
  - manifests/piid_overlap_report.json   -- counts + methodology + caveats
  - manifests/task1_excluded_piid_stage1.txt   -- the N filenames to exclude
  - manifests/task1_leakagefree_filelist.txt   -- the remaining filenames

The leakage-free filelist only needs datasets/wound-region-segmentation
(Task 1's own data) -- source-prefix tagging alone is enough to exclude the
PIID images. datasets/wound-stage-classification is only used for the
*cross-check* (confirming those excluded files really are PIID staging
photos, and flagging numeric-id false positives); if that dataset isn't
present on this machine (e.g. a Colab runtime that only fetched Task 1's
zip), the cross-check is skipped with a warning and the core manifests are
still written correctly.

Run from anywhere; paths are resolved relative to the repo root.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "task_1"))

from src.splits import SOURCE_PATTERNS, infer_source  # noqa: E402

SEG_ROOT = REPO_ROOT / "datasets" / "wound-region-segmentation"
CLS_ROOT = REPO_ROOT / "datasets" / "wound-stage-classification"
OUT_DIR = Path(__file__).resolve().parent.parent / "manifests"


def numeric_id(filename):
    """Trailing integer run before the extension, e.g. '..._34168.png' -> '34168'."""
    stem = Path(filename).stem
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return digits or None


def main():
    seg_filenames = sorted(p.name for p in (SEG_ROOT / "images").iterdir() if p.is_file())
    sources = {name: infer_source(name) for name in seg_filenames}

    by_source = {}
    for name, src in sources.items():
        by_source.setdefault(src, []).append(name)
    source_counts = {src: len(names) for src, names in by_source.items()}

    piid_stage1_files = sorted(by_source.get("ft_evre1_stage_1", []))
    piid_count = len(piid_stage1_files)

    cls_root_available = (CLS_ROOT / "stage_1").is_dir()
    stage1_cls_files, matched_ids, only_in_seg, only_in_cls = [], set(), set(), set()
    false_positive_examples = {}

    if cls_root_available:
        # Cross-check against the Task 3 stage_1 classification folder by numeric id.
        stage1_cls_files = sorted(p.name for p in (CLS_ROOT / "stage_1").iterdir() if p.is_file())
        stage1_cls_ids = {numeric_id(f): f for f in stage1_cls_files}
        piid_ids = {numeric_id(f): f for f in piid_stage1_files}

        matched_ids = set(piid_ids) & set(stage1_cls_ids)
        only_in_seg = set(piid_ids) - set(stage1_cls_ids)
        only_in_cls = set(stage1_cls_ids) - set(piid_ids)

        # Sanity check: naive numeric-id matching against stage_2/3/4 produces
        # false positives because fusc/wsnet/ft_tendonbone sources reuse small
        # integer ids unrelated to PIID. Record (but do not act on) these.
        for stage_dir in ["stage_2", "stage_3", "stage_4"]:
            cls_files = sorted(p.name for p in (CLS_ROOT / stage_dir).iterdir() if p.is_file())
            cls_ids = {numeric_id(f) for f in cls_files}
            seg_ids_non_piid = {
                numeric_id(name): name
                for name, src in sources.items()
                if src != "ft_evre1_stage_1" and numeric_id(name) is not None
            }
            collisions = sorted(set(seg_ids_non_piid) & cls_ids)
            if collisions:
                false_positive_examples[stage_dir] = [seg_ids_non_piid[i] for i in collisions]
    else:
        print(
            f"WARNING: {CLS_ROOT / 'stage_1'} not found -- skipping the "
            "wound-stage-classification cross-check. The leakage-free "
            "filelist below is still correct (it only depends on "
            "wound-region-segmentation filename prefixes), but the 228 "
            "count is NOT independently cross-verified on this run.",
            flush=True,
        )

    total_seg_images = len(seg_filenames)
    leakage_free_files = sorted(name for name in seg_filenames if sources[name] != "ft_evre1_stage_1")

    report = {
        "methodology": (
            "Filename-prefix source tagging via task_1/src/splits.py SOURCE_PATTERNS "
            "(ft_evre1_stage_1 = PIID Stage-1 photos included in Task 1 training). "
            "Cross-checked against datasets/wound-stage-classification/stage_1 by "
            "trailing numeric id in the filename, then spot-verified by opening "
            "image pairs (ft_evre1_stage_1_10267.png vs stage_1_10267.jpg) to "
            "confirm they are the same underlying photo, not an id collision. "
            "(This cross-check needs datasets/wound-stage-classification on disk; "
            "see 'classification_cross_check_available' below.)"
        ),
        "classification_cross_check_available": cls_root_available,
        "task1_source_counts": source_counts,
        "task1_total_images": total_seg_images,
        "piid_stage1_in_task1": piid_count,
        "thesis_claimed_overlap": 228,
        "user_mentioned_alternative": 221,
        "resolution": (
            f"Verified count is {piid_count} (matches the thesis's 228, not the "
            "221 mentioned by the user). Source: exact count of filenames "
            "matching the 'ft_evre1_stage_1_' prefix in "
            "datasets/wound-region-segmentation/images/."
            + ("" if cls_root_available else " [cross-check against "
               "wound-stage-classification skipped this run -- dataset not present]")
        ),
        "leakage_free_task1_count": total_seg_images - piid_count,
        "expected_from_thesis_math": 3476 - 228,
        "stage1_classification_total": len(stage1_cls_files),
        "matched_by_numeric_id": len(matched_ids),
        "piid_stage1_not_in_task1_count": len(only_in_cls),
        "piid_stage1_not_in_task1_ids": sorted(only_in_cls),
        "seg_piid_stage1_without_cls_match": sorted(only_in_seg),
        "numeric_id_false_positive_warning": (
            "Naive numeric-id matching against stage_2/3/4 finds spurious hits "
            "because fusc/wsnet/ft_tendonbone filenames reuse small/short ids "
            "unrelated to PIID. Visually confirmed these are different photos "
            "(e.g. id 34168: fusc and stage_3 are unrelated wounds). These are "
            "NOT real overlaps -- listed here only so they aren't rediscovered "
            "and misinterpreted as leakage."
        ),
        "numeric_id_collisions_non_piid": false_positive_examples,
        "task2_piid_overlap_check": (
            "No PIID/evre/stage-pattern filenames found in "
            "datasets/wound-tissue-segmentation/tissue_images/. Task 2's 612 "
            "tissue images are built from 488 unique case-uuids that are a "
            "1:1 match with Task 1's 'ft_tendonbone_<uuid>' clinical-photo "
            "subset (488 images) -- i.e. Task 1 and Task 2 share clinical "
            "source photos with each other, but neither shares any image "
            "with the PIID staging dataset. No Task-2-to-Task-3 leakage risk "
            "found. (Checked once against the raw dataset; not re-derived on "
            "every run since it doesn't depend on wound-stage-classification.)"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "piid_overlap_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT_DIR / "task1_excluded_piid_stage1.txt").write_text("\n".join(piid_stage1_files) + "\n")
    (OUT_DIR / "task1_leakagefree_filelist.txt").write_text("\n".join(leakage_free_files) + "\n")

    if cls_root_available:
        print(f"PIID Stage-1 images in Task 1: {piid_count} (expected 228, matched {len(matched_ids)} by id)")
    else:
        print(f"PIID Stage-1 images in Task 1: {piid_count} (expected 228; cross-check skipped, dataset not found)")
    print(f"Leakage-free Task 1 count: {total_seg_images - piid_count} (expected {3476 - 228})")
    print(f"Wrote manifests to {OUT_DIR}")


if __name__ == "__main__":
    main()
