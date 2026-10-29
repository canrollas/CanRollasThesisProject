# Exp10 — Boundary-Specific Representation Learning — Sonuç Özeti (28 Ekim 2026)

Colab'da üç mod (`heads`, `attention`, `contrastive`), üç fold, gerçek
eğitim+test tamamlandı. Kaynak: `../results.csv` (ham sayılar),
`../attention_diagnostics_*.json` (attention tanısı, aşağıdaki özetin
kaynağı), ikisi de indirilen `colab_train.ipynb`'nin hücre çıktılarından
kurtarıldı.

## Sonuç: 81.68 geçilmedi

| Mod | Accuracy | Macro-F1 | QWK |
|---|---|---|---|
| **baseline** (task_3, referans, yeniden koşulmadı) | 0.8168 ± 0.0150 | 0.8182 ± 0.0138 | 0.9138 ± 0.0164 |
| `heads` (sınıra özel başlık, attention yok) | 0.8139 ± 0.0132 | 0.8167 ± 0.0122 | 0.9110 ± 0.0115 |
| `attention` (+ öğrenilen mekânsal attention) | 0.8130 ± 0.0102 | 0.8155 ± 0.0103 | 0.9120 ± 0.0105 |
| `contrastive` (+ sınır başına supervised contrastive) | **0.8167 ± 0.0144** | **0.8188 ± 0.0136** | **0.9138 ± 0.0142** |

- **`contrastive`**, baseline'ı istatistiksel olarak ayırt edilemez biçimde
  tutturdu (fark <0.001 acc, <0.0001 qwk — kendi std'sinin çok altında).
  Hem accuracy hem QWK'da en iyi sonuç, ama "baseline'ı geçti" denemez.
- **`heads`** ve **`attention`**, baseline'ın 1 std'si içinde ama sistematik
  olarak hafif altında (~0.3-0.4pp acc).
- Hiçbir mod exp1/exp2/exp3 gibi baseline'ın **anlamlı altına** düşmedi —
  bu bölümdeki (exp1 → exp11) en "zararsız" negatif sonuç.

## Attention tanısı: mimarinin kendi öne sürdüğü risk gerçekleşti

`scripts/diagnose_attention.py`, üç sınırın öğrenilen attention
haritalarını (`A_0`, `A_1`, `A_2`) test setinde ikili kosinüs benzerliği
ve argmax-uyum oranıyla karşılaştırır.

| Mod | Ort. kosinüs (3 çift × 3 fold) | Ort. argmax-uyumu |
|---|---|---|
| `attention` | **0.9964** | 0.1155 |
| `contrastive` | **0.9170** | 0.2334 |

- **`attention` modunda** üç bağımsız attention konvolüsyonu hiçbir
  zorlama olmadan birbirinin neredeyse aynısını öğrendi (kosinüs ~0.99
  her çiftte, her fold'da) — mimarinin kendi README'sinde adı konmuş risk
  ("üç ayrı attention modülü aynı özellikleri öğrenebilir") burada
  doğrulandı.
- **`contrastive` kaybı bunu ölçülebilir şekilde düzeltti**: ortalama
  kosinüs 0.996 → 0.917, argmax-uyumu ~%12 → ~%23 (fold2'de
  `boundary0_vs_boundary1` çifti 0.7453'e kadar düştü — gerçek bir
  ayrışma, gürültü değil). Kayıp tasarlandığı şeyi yapıyor.
- **Ama bu ayrışma doğruluğa dönüşmedi**: `contrastive`'in test skoru
  `attention`'dan 1 std içinde, anlamlı şekilde iyi değil. exp6'nın
  bulgusuyla aynı örüntü — temsil ayrışması gerçek ve ölçülebilir, ama
  CORN'un karar sınırını hiçbir fold'da yeterince hareket ettirmiyor.

## Sonuç

Mimari doğru çalışıyor (baseline-koruma, donuk backbone, gradyan akışı —
hepsi `test_model.py` ve provenance JSON'larında doğrulandı) ve
`contrastive` kaybı iddia ettiği mekanizmayı (sınır temsillerini
ayrıştırma) gerçekten üretiyor — ama bu, task_3'ün 81.68 baseline'ını
geçecek kadar güçlü bir sinyale dönüşmüyor. `heads`/`attention`/`contrastive`
üçü de istatistiksel olarak baseline'dan ayırt edilemez; en iyi okuma
"fark yok", "kötüleşme" değil.
