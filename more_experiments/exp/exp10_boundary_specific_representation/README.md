# Exp10 — Boundary-Specific Representation Learning for CORN

**Durum (28 Ekim 2026):** Üç mod da Colab'da gerçek üç-fold eğitim/test
tamamladı. **81.68 geçilmedi.** `contrastive`, baseline'ı istatistiksel
olarak ayırt edilemez biçimde tutturdu (0.8167 vs 0.8168 acc); `heads` ve
`attention` baseline'ın 1 std'si içinde ama hafif altında kaldı. Attention
tanısı (`scripts/diagnose_attention.py`) mimarinin kendi öne sürdüğü riski
doğrudan doğruladı: `attention` modunda üç sınırın attention haritaları
neredeyse birbirinin aynısı (ortalama kosinüs ≈0.996); `contrastive`
kaybı bunu ölçülebilir şekilde ayrıştırdı (ortalama kosinüs ≈0.917,
argmax-uyumu ~%12'den ~%23'e çıktı) ama bu ayrışma doğruluğa yansımadı.
Ayrıntılar aşağıdaki "Sonuçlar" bölümünde.

## Problem

task_3'ün MiT-B1+CORN modelinde (`task_3/src/model.py`) encoder, görüntüyü
tek bir 512 boyutlu özellik vektörüne indirger
(`Backbone.forward`: `last_feature_map.mean(dim=(2,3))`), ve `CornHead`
bu vektör üzerinde tek bir `Linear(512,3)` katmanıdır. CORN'un üç koşullu
kararı -- "stage 1'den ileri mi?", "stage 2'den ileri mi?", "stage 3'ten
ileri mi?" -- bu nedenle **aynı havuzlanmış temsile** dayanır; yalnızca o
üç kararın doğrusal ağırlık satırları birbirinden farklıdır. Encoder'a
hiçbir sınır için "bu karara özel hangi bölgeye bak" sinyali verilmez.

exp5/6/7'nin tanı sonuçları bu boşluğu dolaylı olarak doğruluyor: üç
deneyde de eklenen ek dal (doku CNN'i, el yapımı doku istatistiği, ROI
local branch) ya neredeyse sıfır içerik-bağımlılığı gösterdi ya da hiçbir
tahmini değiştirmedi -- sinyal zayıf değil, mimari onu CORN'un üç kararına
ayrıştıracak bir mekanizma sunmuyordu.

## Önerilen mekanizma

```mermaid
flowchart LR
    R[RGB 224x224] --> B[Donmuş, eğitilmiş MiT-B1 backbone]
    B --> F["Son özellik haritası F, [B,512,7,7]"]
    F --> G[Donmuş GAP + donmuş baseline CornHead]
    G --> L0[baseline_logit_k, 3 boyut]
    F --> A1[Sınır 0 attention A_0 = sigmoid(Conv1x1)]
    F --> A2[Sınır 1 attention A_1]
    F --> A3[Sınır 2 attention A_2]
    A1 --> Z1["z_0 = GAP(F * A_0)"]
    A2 --> Z2["z_1 = GAP(F * A_1)"]
    A3 --> Z3["z_2 = GAP(F * A_2)"]
    Z1 --> M1[MLP_0] --> D1[delta_logit_0]
    Z2 --> M2[MLP_1] --> D2[delta_logit_1]
    Z3 --> M3[MLP_2] --> D3[delta_logit_2]
    L0 --> S["logit_k = baseline_logit_k + delta_logit_k"]
    D1 --> S
    D2 --> S
    D3 --> S
    S --> C[CORN ordinal median -> stage 1-4]
```

`logit_k = baseline_logit_k + delta_logit_k`. `baseline_logit_k`,
task_3'ün gerçek eğitilmiş `mit_b1__corn` checkpoint'inin (exp5/6/7'nin
yüklediği aynı dosya, `task_3/results/task_3_outputs/checkpoints/
mit_b1__corn_fold{0,1,2}.pt`) donmuş `CornHead`'inden gelir -- her
koşulda, her fold'da aynıdır ve asla eğitilmez. `delta_logit_k` her
sınırın kendi ufak kalıntı (residual) başlığından gelir. Backbone da
Phase A boyunca donar (exp5/6/7 ile aynı kural): yeni başlığın katkısı,
backbone'un kendisindeki bir değişiklikle karışmaz.

Üç mod aynı mimariyi paylaşır (`scripts/model.py`'nin `BoundaryCornHead`'i):

| Mod | Attention (`A_k`) | Test ettiği soru |
|---|---|---|
| `heads` | Sabit 1 (yok) -- her sınır GAP(F)'in aynı vektörünü okur | Yalnızca sınıra özel **doğrusal olmayan karar fonksiyonu** (paylaşılan `Linear(512,3)`'ün 3 bağımsız satırından farklı bir şey) işe yarıyor mu? |
| `attention` | `sigmoid(Conv1x1_k(F))`, öğrenilir | Sınıra özel **mekânsal seçim** (`z_k` artık farklı bölgeleri havuzluyor) işe yarıyor mu? |
| `contrastive` | `attention` ile aynı + sınır başına supervised contrastive terim (`scripts/losses.py`) | Sınır temsillerini açıkça ayrıştırmaya zorlamak (aynı taraf yakın, karşı taraf uzak) işe yarıyor mu? |

`A = task_3`'ün kendi ablation sonucu (0.8168 ± 0.0150 acc), yeniden
eğitilmez; doğrudan referans alınır.

## Baseline-koruma tasarımı ve kasıtlı istisna

exp5/6/7'nin kalıntı (residual) tasarımı son katmanı sıfırla başlatarak
kuruluşta `fused_logits == baseline_logits` eşitliğini garanti eder.
Burada **bilerek farklı bir seçim yapıldı**: `MLP_k`'nin son katmanı
sıfırla başlatılmaz (PyTorch'un olağan ilkleşimi kullanılır). Nedeni şu:
sıfırla başlatılan bir son katman, geri yayılımın İLK adımında yalnızca kendi ağırlığına sıfır
olmayan gradyan alır; ondan ÖNCEKİ katmanlar (ve attention konvolüsyonu)
o adımda hâlâ sıfır üst-gradyan görür (aradaki sıfır ağırlık matrisiyle
çarpıldığı için), bu da attention modülünün öğrenmeye başlamasını
geciktirir. Bedeli açıkça şudur: kuruluşta bu modelin çıktıları donmuş
baseline'ın çıktılarına **tam olarak eşit değildir** (exp5/6/7'nin
aksine). Bu kasıtlıdır, hata değildir -- `scripts/test_model.py` neyin
gerçekten doğru olduğunu doğrular: donmuş baseline alt-hesabı referans
`CornHead` ile birebir eşleşir, backbone hiçbir gradyan almaz (her üç
modda), ve attention konvolüsyonunun ağırlığı tek bir gerçek geri
yayılım adımından sonra sıfır olmayan gradyan alır (her üç modda).
`attention` modunda `A_k`, kuruluşta sabit 0.5'tir (sıfırla başlatılan
konvolüsyon + sigmoid) -- dejenere değildir, sigmoid'in 0'daki türevi
0.25'tir, yani kapı ilk adımdan itibaren öğrenmeye başlar.

## Kritik uyarı -- bu mimari otomatik olarak daha iyi olmayacak

Üç ayrı attention modülü, hiçbir zorlama olmadan birbirinin aynısını
öğrenebilir -- bu durumda mimari, paylaşılan `Linear(512,3)`'ten
ölçülebilir hiçbir farkı olmayan, sadece daha fazla parametreli bir
yeniden parametrelendirme olur. `scripts/diagnose_attention.py` bunu
doğrudan ölçer: eğitilmiş bir checkpoint'in üç sınır attention haritasını
test görüntüleri üzerinde ikili kosinüs benzerliği, Pearson korelasyonu
ve argmax-uyum oranıyla karşılaştırır. Yüksek benzerlik (kosinüs ~1,
yüksek argmax uyumu her üç çiftte) mimarinin önermesinin (sınıra özel
mekânsal seçim) gerçekte gerçekleşmediğini gösterir -- test doğruluğu ne
derse desin. `contrastive` kaybı da özellikle Stage 3-4 arasındaki klinik
belirsizlik nedeniyle performansı düşürebilir; bu sınırda temiz bir ikili
ayrım olmayabilir.

## Aşamalı deney planı

| Deney | Mimari | Durum |
|---|---|---|
| A | `task_3`'ün mevcut MiT-B1+CORN'u | Zaten eğitilmiş, yeniden koşulmuyor (0.8168 ± 0.0150 acc) |
| B (`--mode heads`) | 3 sınıra özel kalıntı başlık, attention yok | **Henüz eğitilmedi** |
| C (`--mode attention`) | B + öğrenilen mekânsal attention | **Henüz eğitilmedi** |
| D (`--mode contrastive`) | C + sınır başına supervised contrastive kayıp | **Henüz eğitilmedi** |

Yalnızca D'nin iyi çıkması yetmez; B ve C de aynı koşullarda
karşılaştırılmalı ki katkının nereden geldiği (karar fonksiyonu mu,
mekânsal seçim mi, temsil ayrıştırması mı) ayrılabilsin.

## Veri ve bölünme

Hiçbir yeni veri, maske veya türetilmiş modalite yok. Orijinal PIID
staging fotoğrafları (`datasets/wound-stage-classification/`), aynı
3-fold bölünme algoritması (`task_3/src/splits.py`'nin `build_folds`'u,
aynı seed=42) -- task_3'ün kendi baseline'ıyla doğrudan, eşleştirilmiş
karşılaştırma, yeni sızıntı yüzeyi yok.

## Sonuçlar (Colab, üç mod, üç fold, gerçek eğitim)

`results/results.csv`, `early_stopped_at_epoch` 11-13 arasında (patience=10),
`backbone_and_baseline_head_unchanged_across_training: true` her fold/modda
(provenance JSON'larında doğrulandı).

| Mod | Accuracy | Macro-F1 | QWK |
|---|---|---|---|
| **baseline** (task_3, referans, yeniden koşulmadı) | 0.8168 ± 0.0150 | 0.8182 ± 0.0138 | 0.9138 ± 0.0164 |
| `heads` | 0.8139 ± 0.0132 | 0.8167 ± 0.0122 | 0.9110 ± 0.0115 |
| `attention` | 0.8130 ± 0.0102 | 0.8155 ± 0.0103 | 0.9120 ± 0.0105 |
| `contrastive` | **0.8167 ± 0.0144** | **0.8188 ± 0.0136** | **0.9138 ± 0.0142** |

(ortalama ± std, 3 fold üzerinde; std popülasyon std'si, `task_3/src/utils.py`'nin
`write_summary`'siyle aynı kural)

**`contrastive`, baseline'dan istatistiksel olarak ayırt edilemez** (fark
<0.001 acc, <0.0001 qwk -- kendi std'sinin çok altında). `heads` ve
`attention` baseline'ın 1 std'si içinde ama sistematik olarak hafif altında.
**Hiçbir mod 81.68'i geçmedi**, ama -- exp1/exp2/exp3'ün aksine -- hiçbiri
baseline'ın anlamlı altına da düşmedi; bu bölümdeki en "zararsız" sonuç.

### Attention tanısı -- mimarinin kendi öne sürdüğü riski doğrudan test eder

`scripts/diagnose_attention.py`, test setinde üç sınırın (`A_0`, `A_1`,
`A_2`) öğrenilen attention haritalarını ikili kosinüs benzerliği, Pearson
korelasyonu ve argmax-uyum oranıyla karşılaştırır (`results/attention_diagnostics_*.json`,
notebook çıktısından kurtarıldı):

| Mod | Ortalama kosinüs (3 çift × 3 fold) | Ortalama argmax-uyumu |
|---|---|---|
| `attention` | **0.9964** | 0.1155 |
| `contrastive` | **0.9170** | 0.2334 |

**`attention` modunda mimarinin kendi öne sürdüğü risk gerçekleşti**: üç
bağımsız attention konvolüsyonu, hiçbir zorlama olmadan birbirinin neredeyse
aynısını öğrendi (kosinüs ~0.99, her üç çiftte ve her fold'da). Mekânsal
seçimin sınıra özel olduğu iddiası bu haliyle desteklenmiyor.

**`contrastive` kaybı bunu kısmen düzeltti**: ortalama kosinüs 0.996'dan
0.917'ye düştü, argmax-uyumu ~%12'den ~%23'e çıktı (fold2'de `boundary0_vs_boundary1`
çifti 0.7453'e kadar düştü -- gerçek bir ayrışma, sadece gürültü değil).
Kayıp, tasarlandığı şeyi yapıyor: sınır temsillerini ölçülebilir şekilde
birbirinden ayrıştırıyor.

**Ama bu ayrışma doğruluğa dönüşmedi** -- `contrastive`'in test skoru
`attention`'dan anlamlı şekilde iyi değil (0.8167 vs 0.8130, 1 std içinde).
exp6'nın bulgusuyla aynı örüntü: hand-crafted doku istatistikleri CNN'e göre
25x daha içerik-duyarlı kalıntı (`delta_z`) üretti ama agregat doğruluğu
hareket ettirmedi. Burada da temsil ayrışması (mimarinin iddia ettiği
mekanizma) gerçekleşti, ama bu ayrışma CORN'un karar sınırını hiçbir fold'da
yeterince değiştirmedi.

## Çalıştırma

```bash
# Mimari testleri (checkpoint gerektirmez)
task_1/.venv/bin/python more_experiments/exp/exp10_boundary_specific_representation/scripts/test_model.py

# Kısa uçtan uca kontrol: 1 fold, 2 epoch, 8 train + 4 val + 4 test
task_1/.venv/bin/python more_experiments/exp/exp10_boundary_specific_representation/scripts/train.py --mode heads --smoke

# Gerçek eğitim: üç mod, aynı üç fold
for mode in heads attention contrastive; do
  task_1/.venv/bin/python more_experiments/exp/exp10_boundary_specific_representation/scripts/train.py --mode "$mode"
done

# Attention tanı: üç sınırın attention haritaları ne kadar farklı?
task_1/.venv/bin/python more_experiments/exp/exp10_boundary_specific_representation/scripts/diagnose_attention.py \
  --mode attention --fold 0 --split test
```

Her mod kendi checkpoint'lerini `checkpoints/mit_b1__corn__boundary_<mode>_fold{0,1,2}.pt`
içine yazar; sonuçlar `results/exp10_results.csv`'ye eklenir
(`config_id`+`fold` anahtarıyla, yeniden çalıştırmada üzerine yazılır).
Her fold için gerekli RGB baseline checkpoint'i yoksa `train.py` eğitime
geçmeden açıkça durur (hangi dosyanın eksik olduğunu söyler) -- sessizce
ImageNet ilkleşimine dönmez.

## Değerlendirme kriteri

Başarı: `attention` veya `contrastive` modunun üç-fold test ortalamasında
task_3'ün 0.8168 ± 0.0150'sini (ve kendi `heads` kontrolünü) geçmesi.
Ama tek başına bu yeterli değildir -- `diagnose_attention.py`'nin pairwise
benzerlik raporu da birlikte sunulmalı: yüksek accuracy ama yüksek
attention-benzerliği, mekanizmanın iddia ettiği şeyi yapmadığını (sadece
fazladan parametre eklemenin işe yaradığını) gösterir. İlk olumlu sonuç
sonrasında farklı eğitim seed'leri ile doğrulama gerekir.
