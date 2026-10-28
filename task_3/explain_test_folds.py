#!/usr/bin/env python3
"""Export original MiT-B1+CORN held-out predictions and auditable explanations.

No training or model selection. Each original image is evaluated only by the
checkpoint whose historical test fold contains it. The report is local and
contains clinical source images; no network access is performed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(ROOT))

from src.dataset import CLASS_ORDER, StageDataset
from src.metrics import compute_metrics, predict_labels
from src.model import build_model
from src.splits import build_folds
from corn_explain import extract_explanations
from explanation_figures import aggregate_figures, render_case, render_threshold_case

DEFAULT_OUTPUT = ROOT / "more_experiments/reports/figures/mit_b1_corn_test_explanations"
PROTOCOL = TASK / "configs/mit_b1_corn_baseline_protocol.json"


def inspect_mask(path, expected_size=None):
    """Read the Task 1 predicted region mask without editor dependencies."""
    with Image.open(path) as image:
        if expected_size is not None and image.size != tuple(expected_size):
            raise ValueError(f"Mask/image size mismatch: {path}")
        raw = np.asarray(image)
        if image.mode in ("L", "I", "I;16"):
            if not np.isin(raw, (0, 1, 2)).all():
                raise ValueError(f"Unknown mask label: {path}")
            return raw.astype(np.uint8, copy=True)
        rgb = np.asarray(image.convert("RGB"))
    labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
    matched = np.zeros(rgb.shape[:2], dtype=bool)
    for index, color in enumerate(((0, 0, 0), (0, 0, 255), (255, 0, 0))):
        locations = np.all(rgb == color, axis=-1)
        labels[locations] = index
        matched |= locations
    if not matched.all():
        raise ValueError(f"Unknown Task 1 mask color: {path}")
    return labels


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows, key="pred_label"):
    result = compute_metrics([r["true_label"] for r in rows], [r[key] for r in rows])
    result["confusion"] = result["confusion"].tolist()
    return result


def region_statistics(cam, labels, prefix="cam_"):
    """Descriptive positive CAM mass / region area; undefined stays null."""
    field = np.asarray(Image.fromarray(np.asarray(cam, np.float32)).resize(
        (labels.shape[1], labels.shape[0]), Image.Resampling.BILINEAR), dtype=np.float64)
    field = np.maximum(field, 0)
    total = float(field.sum())
    result = {}
    for index, name in ((0, "background"), (1, "skin"), (2, "wound")):
        area = float(np.mean(labels == index))
        mass = float(field[labels == index].sum() / total) if total > 0 else None
        result[f"{name}_area_fraction"] = area
        result[f"{prefix}{name}_mass"] = mass
        result[f"{prefix}{name}_enrichment"] = mass / area if mass is not None and area > 0 else None
    return result


def summarize(rows):
    by_fold = {str(fold): metrics([r for r in rows if r["fold"] == fold]) for fold in range(3)}
    fold_stats = {key: {"mean": float(np.mean([v[key] for v in by_fold.values()])),
                        "std": float(np.std([v[key] for v in by_fold.values()]))}
                  for key in ("accuracy", "macro_f1", "mae", "qwk")}
    errors = [r for r in rows if not r["correct"]]
    pairs = []
    for true in range(4):
        for pred in range(4):
            if true != pred:
                subset = [r for r in errors if r["true_label"] == true and r["pred_label"] == pred]
                if subset:
                    pairs.append({"true_stage": true + 1, "pred_stage": pred + 1,
                                  "n": len(subset), "mean_pred_probability": float(np.mean([r["predicted_probability"] for r in subset]))})
    return {"n_images": len(rows), "n_correct": len(rows) - len(errors), "n_errors": len(errors),
            "folds": by_fold, "fold_mean_std": fold_stats, "pooled": metrics(rows),
            "probability_argmax_descriptive": metrics(rows, "prob_argmax"),
            "median_mode_disagreements": sum(r["median_mode_disagree"] for r in rows),
            "errors_with_decoded_class_probability_at_least_0_9": sum(r["predicted_probability"] >= .9 for r in errors),
            "error_pairs": sorted(pairs, key=lambda item: -item["n"]),
            "zero_predicted_last_cams": sum(r["predicted_last_cam_zero"] for r in rows),
            "zero_predicted_middle_cams": sum(r["predicted_middle_cam_zero"] for r in rows)}


def write_analysis(path, stats, rows):
    fm = stats["fold_mean_std"]
    lines = ["# MiT-B1 + CORN: üç fold test açıklamaları", "",
             f"{stats['n_images']:,} farklı test görüntüsü; **{stats['n_correct']} doğru, {stats['n_errors']} hata**.",
             f"Üç-fold accuracy: **%{100 * fm['accuracy']['mean']:.4f} ± {100 * fm['accuracy']['std']:.4f}**; "
             f"macro-F1: %{100 * fm['macro_f1']['mean']:.4f}; QWK: {fm['qwk']['mean']:.6f}.", "",
             "Her görüntü yalnız kendi held-out fold checkpoint'iyle değerlendirilmiştir. Eğitim veya eşik seçimi yapılmadı.",
             "Split görüntü düzeyindedir; hasta bağımsızlığı doğrulanmış değildir.", "",
             "## En sık hatalar", "", "| Gerçek → Tahmin | Görüntü | Ortalama P(verilen karar) |", "|---|---:|---:|"]
    for item in stats["error_pairs"]:
        lines.append(f"| Stage {item['true_stage']} → {item['pred_stage']} | {item['n']} | %{100 * item['mean_pred_probability']:.1f} |")
    lines += ["", f"Yanlış kararına en az %90 olasılık veren hata sayısı: **{stats['errors_with_decoded_class_probability_at_least_0_9']}**.",
              f"CORN kararı ile olasılık argmax'ı **{stats['median_mode_disagreements']}** görüntüde farklıdır. "
              "Raporun ana tahmini tarihsel CORN ordinal-median kararıdır; argmax ayrı bir tanı sütunudur.", "",
              "## Model odağı nasıl okunmalı?", "",
              "Her panel orijinal RGB'yi, verilen kararın orta/son katman Grad-CAM'lerini, gerçek sınıfın son katman haritasını, "
              "dört sınıf olasılığını, üç ham koşullu logiti ve üç kümülatif eşik olasılığını gösterir.",
              "Sınıf hedefi log P(sınıf)'tır; koşullu eşik haritaları ayrı NPZ'de ham logit hedefiyle tutulur.",
              "Son katman 7×7, orta katman 14×14'tür. Görüntü üzerine büyütme yeni mekânsal ayrıntı üretmez. "
              "Renkler harita başına normalize edilir; farklı görüntüler arasında mutlak kanıt büyüklüğü karşılaştırılmaz.",
              "Cilt/yara/arka plan CAM kütlesi, Task 1 modelinin **tahmini** maskeleriyle ölçülür. "
              "Enrichment = CAM kütle payı / bölge alan payı; sıfır CAM veya bulunmayan bölge için tanımsız değer boş bırakılır.", ""]
    for correct, label in ((True, "Doğru"), (False, "Hatalı")):
        group = [r for r in rows if r["correct"] == correct]
        parts = []
        for region in ("wound", "skin", "background"):
            values = [r.get(f"cam_{region}_enrichment") for r in group]
            values = [v for v in values if v is not None]
            if values:
                parts.append(f"{region}: medyan {np.median(values):.2f}× (n={len(values)})")
        if parts:
            lines.append(f"- {label} örneklerde alanla normalize odak: " + "; ".join(parts) + ".")
    lines += ["", "Bu haritalar modelin kararını görsel olarak incelemeye yarayan **kaba, hedefe bağlı açıklamalardır**; "
              "klinik olarak doğru bölgeyi kullandığının veya nedensel bağımlılığın kanıtı değildir. "
              "Özellikle Stage 3↔4 galerilerini ve yüksek güvenli hataları birlikte incelemek yeni validation hipotezleri üretmek içindir. "
              "Test görsellerine göre mimari/karar eşiği seçilirse yeni bağımsız doğrulama gerekir.", "",
              "## Dosyalar", "", "- `index.html`: fold, evre, hata ve güven filtreleriyle yerel inceleme.",
              "- `all_predictions.csv/json`: tüm logitler, koşullu/kümülatif/sınıf olasılıkları ve maske-odak ölçümleri.",
              "- `fold_0/`, `fold_1/`, `fold_2/`: örnek panelleri, sayısal CAM NPZ'leri ve test üyelikleri.",
              "- `summary/`: PNG ve SVG toplu grafikler, hata galerileri.",
              "- `provenance.json`: checkpoint/split/girdi/kod hash'leri ve açıklama yöntemi.", "",
              "Kaynaklar: [Grad-CAM](https://arxiv.org/abs/1610.02391), "
              "[Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292), "
              "[CORN decoder](https://raschka-research-group.github.io/coral-pytorch/api_subpackages/coral_pytorch.dataset/).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


HTML_TEMPLATE = r'''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiT-B1 + CORN · Test incelemesi</title><style>
body{margin:0;background:#101723;color:#e5edf7;font:15px system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:28px}
h1{font-size:28px;margin:0 0 8px}p{line-height:1.65;color:#b5c5d9}a{color:#82d7cd}button,select,input{font:inherit;background:#1b2a3d;color:#eaf3fc;border:1px solid #435771;border-radius:7px;padding:9px}
.tools{display:flex;gap:12px;flex-wrap:wrap;align-items:center;position:sticky;top:0;padding:14px;background:#101723f4;z-index:1}.tools label{display:flex;gap:7px;align-items:center}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:18px}.card{background:#182536;border:1px solid #304761;border-radius:10px;overflow:hidden}.card p{padding:0 14px;margin:9px 0}.card img{display:block;width:100%;height:auto}.bad{color:#ffb08e}.good{color:#89dbb6}
.nav{display:flex;align-items:center;gap:15px;margin:20px 0}.summary{display:flex;gap:15px;flex-wrap:wrap;margin:20px 0}.summary img{max-width:500px;width:100%;border-radius:7px}
@media(max-width:600px){main{padding:15px}.grid{grid-template-columns:1fr}.tools{position:static}}
</style><main><h1>MiT-B1 + CORN · Üç fold test incelemesi</h1><p>__INTRO__</p>
<p><a href="analysis.md">Analiz notu</a> · <a href="all_predictions.csv" download>Tüm sayısal çıktılar (CSV)</a> · <a href="provenance.json">Kaynak ve yöntem kaydı</a></p>
<details><summary>Toplu grafikler</summary><div class="summary">__SUMMARY__</div></details>
<div class="tools"><label>Örnek <select id="status"><option value="wrong">Hatalar</option><option value="all">Tümü</option><option value="correct">Doğrular</option><option value="disagree">CORN / argmax farklı</option></select></label>
<label>Fold <select id="fold"><option value="all">Tümü</option><option>0</option><option>1</option><option>2</option></select></label>
<label>Gerçek <select id="truth"><option value="all">Tümü</option><option value="0">Stage 1</option><option value="1">Stage 2</option><option value="2">Stage 3</option><option value="3">Stage 4</option></select></label>
<label>Tahmin <select id="pred"><option value="all">Tümü</option><option value="0">Stage 1</option><option value="1">Stage 2</option><option value="2">Stage 3</option><option value="3">Stage 4</option></select></label>
<label>Sırala <select id="sort"><option value="confidence">Karar olasılığı ↓</option><option value="entropy">Belirsizlik ↓</option><option value="path">Dosya adı</option></select></label><input id="search" placeholder="Dosya ara…"></div>
<p>Haritalar hedefe bağlı, kaba Grad-CAM açıklamalarıdır. Renkler her harita içinde normalize edilir. Sınıf kararında CORN median kuralı korunur; argmax ayrıca gösterilir.</p>
<div class="nav"><button id="prev">← Önceki</button><span id="count"></span><button id="next">Sonraki →</button></div><div class="grid" id="grid"></div>
<script>const rows=__DATA__;let page=0;const size=24;const $=id=>document.getElementById(id);
function render(){let data=rows.filter(r=>($('status').value==='all'||($('status').value==='wrong'&&!r.correct)||($('status').value==='correct'&&r.correct)||($('status').value==='disagree'&&r.median_mode_disagree))&&($('fold').value==='all'||r.fold===+$('fold').value)&&($('truth').value==='all'||r.true_label===+$('truth').value)&&($('pred').value==='all'||r.pred_label===+$('pred').value)&&r.path.toLowerCase().includes($('search').value.toLowerCase()));
data.sort((a,b)=>$('sort').value==='path'?a.path.localeCompare(b.path):$('sort').value==='entropy'?b.entropy-a.entropy:b.predicted_probability-a.predicted_probability);
page=Math.max(0,Math.min(page,Math.max(0,Math.ceil(data.length/size)-1)));$('count').textContent=`${data.length} örnek · ${data.length?page*size+1:0}–${Math.min((page+1)*size,data.length)}`;$('prev').disabled=page===0;$('next').disabled=(page+1)*size>=data.length;$('grid').replaceChildren();
for(const r of data.slice(page*size,(page+1)*size)){const card=document.createElement('article');card.className='card';const title=document.createElement('p');title.textContent=`Fold ${r.fold} · ${r.path}`;const detail=document.createElement('p');detail.className=r.correct?'good':'bad';detail.textContent=`Gerçek Stage ${r.true_label+1} → CORN Stage ${r.pred_label+1} · P(karar) %${(r.predicted_probability*100).toFixed(1)} · argmax Stage ${r.prob_argmax+1}`;const link=document.createElement('a');link.href=r.matplotlib_error_path||r.case_path;link.target='_blank';const image=document.createElement('img');image.loading='lazy';image.src=link.href;image.alt=`${r.path}: logitler, olasılıklar ve Grad-CAM`;link.append(image);const extra=document.createElement('p');const threshold=document.createElement('a');threshold.href=r.threshold_case_path;threshold.target='_blank';threshold.textContent='Üç CORN eşiğinin odak haritaları';extra.append(threshold);card.append(title,detail,link,extra);$('grid').append(card);}}
for(const id of ['status','fold','truth','pred','sort','search'])$(id).addEventListener(id==='search'?'input':'change',()=>{page=0;render()});$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};render();</script></main></html>'''


def write_gallery(path, rows, stats, figures):
    keep = ("path", "fold", "true_label", "pred_label", "prob_argmax", "correct",
            "median_mode_disagree", "predicted_probability", "entropy", "case_path", "matplotlib_error_path", "threshold_case_path")
    data = [{key: row[key] for key in keep} for row in rows]
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    intro = (f"{len(rows)} farklı held-out test görüntüsü · {stats['n_correct']} doğru / {stats['n_errors']} hata · "
             f"Üç-fold accuracy %{stats['fold_mean_std']['accuracy']['mean']*100:.4f}. "
             "Her örnek yalnız kendi test fold modeliyle hesaplandı. Görsele tıklayarak tam boy açabilirsin.")
    thumbnails = []
    for figure in figures:
        name = figure["path"] if isinstance(figure, dict) else str(figure)
        if name.endswith(".png") and "gallery" not in name:
            source = "summary/" + name
            thumbnails.append(f'<a href="{html.escape(source)}"><img loading="lazy" src="{html.escape(source)}" alt="{html.escape(Path(name).stem)}"></a>')
    path.write_text(HTML_TEMPLATE.replace("__INTRO__", html.escape(intro)).replace("__SUMMARY__", "".join(thumbnails)).replace("__DATA__", encoded), encoding="utf-8")


def run(args):
    torch.set_num_threads(args.threads)
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Use a fresh output directory: {destination}")
    protocol = json.loads(PROTOCOL.read_text())
    folds = build_folds(args.data_root, 3, .1, 42)
    test_paths = [relative for fold in folds for relative, _ in fold["test"]]
    if len(test_paths) != 1091 or len(set(test_paths)) != 1091:
        raise ValueError("Historical test folds must partition the 1,091 source images exactly once")
    checkpoint_info, inputs = {}, {}
    for index, fold in enumerate(folds):
        split_sha = hashlib.sha256(json.dumps(fold, sort_keys=True).encode()).hexdigest()
        checkpoint = args.checkpoint_dir / f"mit_b1__corn_fold{index}.pt"
        checkpoint_sha = sha256(checkpoint)
        expected = protocol["folds"][str(index)]
        if split_sha != expected["split_sha256"] or checkpoint_sha != expected["baseline_sha256"]:
            raise ValueError(f"Fold{index} split/checkpoint differs from the original baseline")
        checkpoint_info[str(index)] = {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha,
                                       "split_sha256": split_sha, "n_test": len(fold["test"])}
    for relative in test_paths:
        image = args.data_root / relative
        mask = args.mask_root / Path(relative).parent / "masks" / (Path(relative).stem + ".png")
        if not mask.is_file():
            raise FileNotFoundError(f"Predicted region mask missing: {mask}")
        inputs[relative] = {"rgb_sha256": sha256(image), "predicted_mask_sha256": sha256(mask)}
    destination.mkdir(parents=True)
    began = time.time()
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "complete": False, "model": "original MiT-B1 + CORN", "training_performed": False,
                  "test_policy": "one original image, one held-out-fold checkpoint; no augmentation or ensemble",
                  "split_unit": "image; patient independence unverified", "checkpoints": checkpoint_info,
                  "source_rgb_root": str(args.data_root.resolve()), "predicted_mask_root": str(args.mask_root.resolve()),
                  "input_hashes": inputs, "device": args.device, "threads": args.threads,
                  "probability_semantics": {"logits": "three conditional binary logits, NOT four nominal class logits",
                    "conditional": "sigmoid(z_k)=P(Y>k | Y>k-1), Y is zero based",
                    "survival": "S_k=product_{j=0..k}sigmoid(z_j)=P(Y>k)",
                    "class_probabilities": "[1-S0,S0-S1,S1-S2,S2]",
                    "decoder": "count(S_k>0.5), exactly coral_pytorch corn_label_from_logits",
                    "probability_argmax": "separate descriptive output; does not replace historical prediction",
                    "entropy": "natural logarithms, nats", "expected_stage": "sum_{c=0..3}(c+1)*P(Y=c)"},
                  "mask_semantics": "leakage-free Task1 predicted regions, not clinical ground truth",
                  "cam_semantics": "ReLU Grad-CAM; target logP(decoded class), logP(true class), or raw conditional threshold logit; per-map max normalization; nonpositive maps flagged",
                  "legacy_gradcam_difference": "old gradcam_mit_b1.py selected probability argmax; this report targets actual CORN decoded class",
                  "code_sha256": {path.name: sha256(path) for path in
                     (Path(__file__), TASK / "corn_explain.py", TASK / "explanation_figures.py",
                      TASK / "src/model.py", TASK / "src/dataset.py", TASK / "src/metrics.py", TASK / "src/splits.py")}}
    json_write(destination / "provenance.json", provenance)
    all_rows = []
    for fold_index, fold in enumerate(folds):
        print(f"fold{fold_index}: {len(fold['test'])} held-out images", flush=True)
        model = build_model("mit_b1", None, "corn", 4)
        checkpoint = args.checkpoint_dir / f"mit_b1__corn_fold{fold_index}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.to(args.device).eval()
        dataset = StageDataset(fold["test"], args.data_root, 224, None)
        batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        fold_rows, offset = [], 0
        for images, targets in batches:
            explained = extract_explanations(model, images.to(args.device), targets=targets.to(args.device), include_thresholds=True)
            decoded = explained["decoded"].numpy()
            np.testing.assert_array_equal(decoded, predict_labels(explained["logits"], "corn").numpy())
            probs = explained["class_probabilities"].numpy()
            for index in range(len(images)):
                relative, true = fold["test"][offset + index]
                pred = int(decoded[index])
                original_path = args.data_root / relative
                with Image.open(original_path) as source:
                    native_rgb = np.asarray(source.convert("RGB"))
                    rgb = np.asarray(source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
                mask_path = args.mask_root / Path(relative).parent / "masks" / (Path(relative).stem + ".png")
                labels = inspect_mask(mask_path, (native_rgb.shape[1], native_rgb.shape[0]))
                case_path = Path(f"fold_{fold_index}/cases") / Path(relative).with_suffix(".png")
                threshold_path = Path(f"fold_{fold_index}/threshold_cases") / Path(relative).with_suffix(".png")
                cam_path = Path(f"fold_{fold_index}/cams") / Path(relative).with_suffix(".npz")
                row = {"path": relative, "fold": fold_index, "true_label": int(true), "pred_label": pred,
                       "true_stage": CLASS_ORDER[true], "pred_stage": CLASS_ORDER[pred],
                       "prob_argmax": int(explained["probability_argmax"][index]), "correct": bool(pred == true),
                       "median_mode_disagree": bool(pred != int(explained["probability_argmax"][index])),
                       "predicted_probability": float(probs[index, pred]), "true_probability": float(probs[index, true]),
                       "max_probability": float(probs[index].max()),
                       "entropy": float(-(probs[index] * np.log(np.maximum(probs[index], 1e-30))).sum()),
                       "expected_stage": float((probs[index] * np.arange(1, 5)).sum()),
                       "case_path": case_path.as_posix(), "cam_path": cam_path.as_posix(),
                       "threshold_case_path": threshold_path.as_posix(),
                       "matplotlib_error_path": (Path(f"fold_{fold_index}/matplotlib_errors") / Path(relative).with_suffix(".png")).as_posix() if pred != true else None}
                for key, values in (("logit", explained["logits"]),
                                    ("conditional", explained["conditional_probabilities"]),
                                    ("survival", explained["survival_probabilities"]),
                                    ("prob", explained["class_probabilities"])):
                    row.update({f"{key}_{j}": float(value) for j, value in enumerate(values[index])})
                panel_cams, arrays = {}, {}
                for target, layers in explained["cams"].items():
                    panel_cams[target] = {}
                    arrays[f"{target}_target_index"] = explained["target_classes"][target][index].numpy()
                    for layer, values in layers.items():
                        panel_cams[target][layer] = values["normalized"][index].numpy()
                        for kind in ("raw", "signed", "normalized", "zero"):
                            arrays[f"{target}_{layer}_{kind}"] = values[kind][index].numpy()
                        row[f"{target}_{layer}_cam_zero"] = bool(values["zero"][index])
                row.update(region_statistics(panel_cams["predicted"]["last"], labels))
                row.update(region_statistics(panel_cams["predicted"]["middle"], labels, "middle_cam_"))
                arrays.update({"logits": explained["logits"][index].numpy(),
                               "class_probabilities": probs[index], "true_label": np.array(true),
                               "pred_label": np.array(pred), "image_shape": np.array(native_rgb.shape[:2])})
                (destination / cam_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination / cam_path, **arrays)
                render_case(rgb, row, panel_cams, destination / case_path)
                render_threshold_case(rgb, row, panel_cams, destination / threshold_path)
                fold_rows.append(row)
            offset += len(images)
            if offset % 25 < len(images) or offset == len(dataset):
                print(f"  fold{fold_index} {offset}/{len(dataset)} · total elapsed {time.time()-began:.0f}s", flush=True)
        expected_correct = (294, 303, 294)[fold_index]
        if sum(r["correct"] for r in fold_rows) != expected_correct:
            raise AssertionError(f"Fold{fold_index} no longer reproduces original baseline correct count")
        if sha256(checkpoint) != checkpoint_info[str(fold_index)]["sha256"]:
            raise ValueError("Checkpoint changed during analysis")
        folder = destination / f"fold_{fold_index}"
        json_write(folder / "test_membership.json", [{"path": p, "label": y} for p, y in fold["test"]])
        json_write(folder / "predictions.json", fold_rows)
        csv_write(folder / "predictions.csv", fold_rows)
        all_rows.extend(fold_rows)
        json_write(destination / "all_predictions.json", all_rows)
        json_write(folder / "explanation_method.json", explained["metadata"])
        del model
    csv_write(destination / "all_predictions.csv", all_rows)
    stats = summarize(all_rows)
    json_write(destination / "metrics.json", stats)
    plots = aggregate_figures(all_rows, destination / "summary")
    from error_example_figures import render_error_examples
    error_figures = render_error_examples(all_rows, destination)
    if len(error_figures) != stats["n_errors"]:
        raise AssertionError("Every error must have its own Matplotlib figure")
    plots["matplotlib_error_examples"] = error_figures
    provenance["code_sha256"]["error_example_figures.py"] = sha256(TASK / "error_example_figures.py")
    json_write(destination / "figure_manifest.json", plots)
    write_analysis(destination / "analysis.md", stats, all_rows)
    write_gallery(destination / "index.html", all_rows, stats, plots["figures"])
    for relative, expected in inputs.items():
        mask = args.mask_root / Path(relative).parent / "masks" / (Path(relative).stem + ".png")
        if sha256(args.data_root / relative) != expected["rgb_sha256"] or sha256(mask) != expected["predicted_mask_sha256"]:
            raise ValueError("Input image/mask changed during analysis")
    provenance["complete"] = True
    provenance["n_test_images"] = len(all_rows)
    provenance["elapsed_s"] = time.time() - began
    json_write(destination / "provenance.json", provenance)
    print(json.dumps({"directory": str(destination), "n_images": len(all_rows),
                      "fold_accuracy_mean": stats["fold_mean_std"]["accuracy"]["mean"],
                      "elapsed_s": provenance["elapsed_s"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/wound-stage-classification")
    parser.add_argument("--mask-root", type=Path, default=ROOT / "datasets/wound-staging-masks")
    parser.add_argument("--checkpoint-dir", type=Path, default=TASK / "results/task_3_outputs/checkpoints")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
