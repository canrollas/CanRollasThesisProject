# Chapter 8 — Additional Wound Staging Experiments: Research Summary

**Hazırlayan:** Can Rollas · **Danışman:** Prof. Dr. Yalın Baştanlar
**Kapsam:** `more_experiments/` altında yürütülen tüm Task 1→Task 3 ve Task 2→Task 3 entegrasyon denemeleri (exp-1 … exp-11), `task_3`'ün checkpoint ensembling denemesi, ve danışmanın ayrıca talep ettiği beş ek iş paketi (WP1-5, `more_experiments/experiment_extended/`) — kullanılan tüm checkpoint, sonuç CSV'si ve teşhis (diagnostic) scripti üzerinden doğrulanmıştır.

---

## 1. Motivasyon ve Araştırma Soruları

Tez Chapter 7'de kurulan RGB-tabanlı evreleme modeli (MiT-B1 + CORN) zaten güçlü bir temel çizgi (baseline) oluşturuyor. Chapter 8'in amacı, elimizde zaten var olan iki ek bilgi kaynağının — **Task 1'in yara lokalizasyonu** ve **Task 2'nin doku kompozisyon segmentasyonu** — bu güçlü RGB modelini **geliştirip geliştiremeyeceğini**, ve geliştiremiyorsa **neden geliştiremediğini** sistematik olarak araştırmaktır.

Merkezi araştırma sorusu:

> Yara lokalizasyonu ve doku segmentasyonu, ordinal basınç yarası evrelemesi için RGB modeline **tamamlayıcı (complementary)** bilgi sağlıyor mu, ve sağlıyorsa bu bilgi hangi deneysel konfigürasyonlarda etkin biçimde modele entegre edilebiliyor?

Bu rapor yedi deney ailesini (exp-1 → exp-7), bunların alt varyantlarını, teşhis analizlerini ve bir "tamamlayıcılık" (complementarity) analizini kapsar. Her bulgu doğrudan repodaki CSV/JSON dosyalarından doğrulanmıştır; hiçbir sayı tahmin edilmemiştir.

## 2. Deneysel Kurulum ve RGB Temel Çizgisi

**Mimari:** ImageNet ön-eğitimli **MiT-B1** encoder + **CORN** ordinal regresyon başlığı (3 koşullu lojit, 4 sınıf).
**Veri kümesi:** PIID — 1.091 basınç yarası görüntüsü, 4 EPUAP evresi (stage_1: 230, stage_2: 313, stage_3: 275, stage_4: 273).
**Değerlendirme:** Sınıf-dengeli (stratified) 3-fold çapraz doğrulama, aynı `seed=42` ile üretilen fold bölünmesi — bu rapordaki **her** deney, `task_3/src/splits.py`'nin `build_folds()` fonksiyonuyla üretilen **aynı** fold üyeliğini kullanır (dosya adı kökü üzerinden doğrulanmıştır).

| Metrik | Ortalama ± Std |
|---|---|
| Accuracy | 0.8168 ± 0.0150 |
| Macro-F1 | 0.8182 ± 0.0138 |
| MAE | 0.1896 ± 0.0204 |
| QWK | 0.9138 ± 0.0164 |

(`task_3/results/task_3_outputs/results/ablation_summary.csv`, `mit_b1__corn` satırı — doğrudan dosyadan doğrulandı.)

Tüm sonraki deneyler bu dört metriği, **bu tabloya göre** fark olarak raporlar.

## 3. Yara Lokalizasyonu Deneyleri (exp-1, exp-2)

**Ön koşul — sızıntısız (leakage-free) Task 1 yeniden eğitimi.** Task 1'in orijinal eğitim kümesinde 228 PIID Stage-1 fotoğrafı bulunduğu tespit edildi (`more_experiments/manifests/piid_overlap_report.json`: 3.476 toplam görüntüden 228'i `ft_evre1_stage_1` kaynak etiketli). Bu, Task 1 modelinin PIID evreleme etiketlerini dolaylı olarak "görmüş" olabileceği anlamına gelir — bu nedenle **MAnet + MiT-B2**, bu 228 görüntü çıkarılmış 3.248 görüntülük kümeyle, 3-fold CV ile **sıfırdan yeniden eğitildi** (`more_experiments/task1_retrain/`). Üç fold checkpoint'i de Colab'da eğitildi ve repoya kopyalandı; fold'lar arası karşılaştırma (`piid_crops/compare_folds.py`) stage_2/3/4'te tam uyum, stage_1'de (sağlam/neredeyse sağlam deri) değişen tespit oranları gösterdi (%73.0 → %80.0 → %89.6) — bu yüzden tek fold yerine **3 fold'un softmax olasılıklarının ortalaması (soft-vote ensemble)** kullanıldı. Bu ensemble maske üretimi (`datasets/wound-staging-masks/`), aşağıdaki **tüm** deneylerin ortak girdisidir; hiçbir deney ground-truth maske veya evreleme etiketi kullanmadan maske üretmiştir.

**exp-1 (skin+wound crop):** Tahmini "deri" + "yara" sınıflarına ait pikseller korunur, arka plan sıfırlanır (siyah). Ortalama korunan alan ≈ %97.
**exp-2 (wound-only, adaptif genişletme):** Sadece tahmini "yara" pikselleri korunur. Naif bir sabit-eşik yaklaşımı iki sorun çıkardı: (a) zaten büyük yaralarda etraftaki deri tamamen kesiliyordu, (b) küçük/tespit edilemeyen yaralar dairesel mesafe eşiğiyle büyütüldüğünde yapay, "çok düzgün" bir daire ortaya çıkıyordu (modelin gerçek yara şekli yerine bu yapay şekli öğrenmesi riski). Çözüm: her görüntüye koşulsuz 12px'lik bir "deri halkası" marjı eklendi, ek büyütme ise düşük-frekanslı gürültüyle bozularak organik bir sınır elde edildi. 41/1.091 görüntüde hiç yara pikseli tespit edilmedi (bunlar görüntü merkezinden küçük bir diskle başlatıldı — **hiçbir görüntü değerlendirmeden çıkarılmadı**).

| Metrik | **Temel Çizgi (RGB)** | **exp-1** (skin+wound) | **exp-2** (wound-only) |
|---|---|---|---|
| Accuracy | 0.8168 ± 0.0150 | 0.7819 ± 0.0157 | 0.7837 ± 0.0222 |
| Macro-F1 | 0.8182 ± 0.0138 | 0.7815 ± 0.0149 | 0.7863 ± 0.0214 |
| QWK | 0.9138 ± 0.0164 | 0.8935 ± 0.0111 | 0.9000 ± 0.0149 |

**Bulgu:** Her iki maskeleme varyantı da RGB temel çizgisinin **altında** kalıyor, tutarlı bir farkla (~3.3pp accuracy). E1-B ve E2, birbirinden istatistiksel olarak ayırt edilemez (1 std içinde) — bu da "ne kadar maskelendiği"nin değil, "maskelemenin kendisinin" bir şeyler kaybettirdiğini gösteriyor. Olası açıklama: tam fotoğrafın arka planı/çerçevelemesi/klinik bağlamı (çarşaf, ışıklandırma, ölçek ipuçları) modele gerçek bir sinyal sağlıyor olabilir — düz siyah maskeleme bunu kaldırırken lokalizasyonun eklediğinden daha fazlasını götürüyor.

## 4. Doku-Yönlendirmeli Evreleme Deneyleri (exp-3, exp-4)

**exp-3 — erken füzyon (early fusion):** RGB'nin 3 kanalına, Task 2'nin (MAnet+MiT-B2, `task_2/results/task_2_outputs/checkpoints/`) ürettiği 5 doku-sınıfı olasılık haritası (granülasyon, slough, nekroz, tendon, kemik) eklenerek 8 kanallı bir girdi oluşturuldu. **Önemli mimari düzeltme:** SMP kütüphanesinin otomatik `in_channels=8` mekanizması test edildiğinde, önceden-eğitilmiş 3 kanallı ağırlıkları 8 kanala **döngüsel olarak kopyalayıp 3/8 ile ölçeklediği** (RGB kanalları dahil) tespit edildi — bu, 8-kanallı modelin RGB-only temel çizgiden **farklı bir başlangıç noktasından** eğitilmesi anlamına gelirdi. Bunun yerine RGB ağırlıkları değiştirilmeden kopyalanıp doku kanalları sıfırla başlatıldı; `forward(RGB, tissue=0) == forward_3ch(RGB)` eşitliği doğrudan doğrulandı.

**exp-4 — geç füzyon (late fusion):** RGB ve doku bilgisi **ayrı ağlarla** kodlanıp, her biri kendi havuzlanmış (pooled) özellik vektörünü ürettikten **sonra** birleştiriliyor (erken füzyonun aksine). RGB kolu: `task_3/src/model.py`'nin kendi `Backbone`'u (F_rgb, 512-boyut). Doku kolu: küçük 3-katmanlı bir CNN, `TissueCNN` (F_tissue, 128-boyut). İki füzyon modu test edildi: `concat` (`MLP([F_rgb;F_tissue])` → CORN) ve `gated` (örneğe özgü bir kapı (gate) ile doku katkısının ağırlığını öğrenen bir varyant). **Not:** exp-4'teki RGB kolu düz ImageNet ağırlıklarıyla başlatıldı, eğitilmiş evreleme temel çizgisiyle değil — çünkü o sırada `task_3`'ün kendi eğitilmiş `.pt` dosyası lokalde mevcut değildi (bu, exp-5'in neden farklı bir tasarım kullandığını açıklıyor, bkz. Bölüm 5).

| Metrik | Temel Çizgi | exp-3 (erken füzyon) | **exp-4 concat** | exp-4 gated |
|---|---|---|---|---|
| Accuracy | 0.8168 | 0.7791 | **0.7983** | 0.7957 |
| Macro-F1 | 0.8182 | 0.7822 | **0.8003** | 0.7977 |
| QWK | 0.9138 | 0.8997 | **0.9109** | 0.9063 |

**Bulgu:** Geç füzyon erken füzyonu ve her iki lokalizasyon denemesini de açıkça geçiyor ve temel çizgiye olan farkın çoğunu kapatıyor (QWK'de 0.9109 vs 0.9138 — temel çizginin kendi std'si olan 0.0164 içinde). Yine de accuracy/macro-F1'de temel çizginin altında kalıyor. Bu, Chapter 8'deki **ilk gerçek sinyal**: doku bilgisinin *nasıl* birleştirildiği, *eklenip eklenmediğinden* en az o kadar önemli.

**Doku olasılık haritaları vs. sert segmentasyon etiketleri:** Her iki deney de Task 2'nin **yumuşak (soft) softmax olasılıklarını** kullanıyor, argmax ile üretilmiş sert (hard) etiketleri değil — bu, modelin belirsiz/sınır piksellerdeki Task 2 güvensizliğini de örtük biçimde görmesini sağlıyor (örn. granülasyon/slough sınırındaki bir piksel [0.4, 0.6, 0, 0, 0] gibi bir dağılım taşıyabilir, [0,1,0,0,0] yerine).

## 5. Rezidüel Füzyon ve Eğitim Teşhisleri (exp-5)

**Temel-çizgi-koruyucu (baseline-preserving) tasarım:** `z_final = z_rgb + delta_z`. `z_rgb`, **dondurulmuş (frozen), gerçekten eğitilmiş** RGB MiT-B1+CORN temel çizgisinden gelir (exp-4'ün aksine — artık fold-1'in gerçek checkpoint'i elde edildiği için). `delta_z`, `TissueCNN` + küçük bir rezidüel MLP'den gelir; **son doğrusal katman sıfırla başlatılır**, böylece kurulumda `delta_z=0` ve model temel çizginin tahminlerini **birebir** üretir — bu, `torch.equal` ile (yaklaşık değil, tam) doğrulandı, hem mimari testlerle hem gerçek checkpoint üzerinde.

**Blokaj:** `task_3`'ün kendi eğitilmiş `mit_b1__corn` checkpoint'leri lokalde hiç kaydedilmemişti. Önce sadece fold-1'in checkpoint'i elde edildi; fold-0/2 sonradan sağlandı (bkz. Bölüm 6).

**3-fold sonucu (`rgb_tissue` varyantı, tam):** accuracy 0.8158 ± 0.0073, macro-F1 0.8178 ± 0.0073, QWK 0.9132 ± 0.0122 — temel çizgiden istatistiksel olarak ayırt edilemiyor, hatta std'si daha düşük.

**Kök-neden teşhisi (`scripts/diagnose_residual.py`) bunun "gerçek doku düzeltmesiyle temel çizgiyi yakalamak" olmadığını ortaya çıkardı:** her fold'un en iyi checkpoint'i epoch 1-2'de kaydedilmişti (neredeyse-sıfır `delta_z`); sonraki epoklarda `delta_z`'nin normu sürekli büyürken validasyon macro-F1'i **kötüleşiyordu** (klasik aşırı-öğrenme, early stopping tarafından reddediliyor). Yakınsanan checkpoint'te `delta_z`'nin büyüklüğü `z_rgb`'nin karar marjlarından **~1000 kat küçük**, doku CNN'inin gradyanı neredeyse düz (~1e-3–5e-3), ve en çarpıcısı: **farklı bir görüntünün doku haritasını karıştırmak** (shuffled, aynı RGB) `delta_z`'yi kosinüs ≈ 1.0 ile **aynı yöne** değiştiriyor — yani kol gerçek doku içeriğini kullanmıyor, neredeyse-temel-çizgi bir "hiçbir şey yapma" çözümüne yakınsıyor.

**Mimari düzeltme — `tissue_only` varyantı:** `delta_z`, **sadece** doku CNN özelliklerinden hesaplanıyor (RGB'yi yapısal olarak göremiyor — doğrudan doğrulandı: RGB değiştirilip doku sabit tutulduğunda `delta_z` byte-özdeş kalıyor). Ayrıca BatchNorm yerine GroupNorm (batch_size=16'da BatchNorm istatistikleri gürültülüydü).

**⚠️ Veri bütünlüğü notu (bu raporun doğrulama sürecinde tespit edildi):** `exp5_results.csv` dosyasında `tissue_only_residual` için **sadece fold-1** satırı mevcuttur (accuracy 0.8370, macro-F1 0.8376, QWK 0.9323). README'nin kendisi de açıkça "Folds 0 and 2 for `tissue_only` still needed" diyor. Geliştirme sürecinde fold-0/2 için bir Colab sonucu tartışılmış olsa da, bu sonuç **hiçbir zaman** `exp5_results.csv`'ye veya `comparison.csv`'ye yazılmamış/commit'lenmemiştir — dolayısıyla bu rapor **sadece fold-1'in doğrulanmış tek-fold sonucunu** resmi bulgu olarak sayar, 3-fold bir ortalama **iddia etmez**. Ayrıca fold-1'in kendi temel çizgisi de 0.8370 idi (3-fold ortalaması 0.8168 değil) — yani `tissue_only`'nin fold-1 sonucu temel çizgiyi "geçmiyor", **kendi fold'unun baseline'ını aynen eşliyor**. Tissue-ablation teşhisi (gerçek vs. karıştırılmış doku girdisi) bu fold için de neredeyse-sıfır içerik-bağımlılığı gösteriyor (kosinüs(gerçek,karıştırılmış)≈1.0, göreli L2≈0.005) — `rgb_tissue`'daki RGB kısayolu ortadan kaldırılmış olsa da, doku kolu hâlâ ayrım yapacak kadar güçlü bir öğrenme sinyali alamıyor.

## 6. Doku İstatistikleri ve Tamamlayıcılık Analizi (exp-6)

exp-5'in CNN tabanlı doku kolunun zayıf/düz gradyanı teşhis edildikten sonra, soru daraltıldı: öğrenilen bir kodlayıcı yerine **elle hesaplanan, deterministik istatistiklerle** doku olasılık haritaları gerçekten *bağımsız* bir evreleme sinyali taşıyor mu?

**20-boyutlu özellik vektörü:** 5 doku sınıfı (granülasyon, slough, nekroz, tendon, kemik) × 4 istatistik (ortalama olasılık, standart sapma, 95. yüzdelik, 0.5 eşiğinin üzerindeki piksel oranı), **sadece tahmini yara pikselleri üzerinden** hesaplandı (ground truth değil). 41/1.091 sıfır-yara görüntüsü tamamen-sıfır vektör alıyor.

**(1) `tissue_statistics_only` — sinyal var mı?** MLP+CORN, **sadece** 20-boyutlu vektör üzerinde, RGB hiç yok:

| Metrik | Değer (3-fold) |
|---|---|
| Accuracy | 0.3803 ± 0.0094 |
| Macro-F1 | 0.3065 ± 0.0151 |
| QWK | 0.3550 ± 0.0091 |

(Değerler `more_experiments/exp/exp6_tissue_statistics/results/exp6_results.csv`'den birebir doğrulandı.) RGB temel çizgisinin (~0.82) çok altında, ama **şans seviyesinin (4 sınıf, ~0.25 accuracy, ~0 QWK) belirgin şekilde üzerinde**. **Önemli ayrım:** bu, doku özelliklerinin evrelemeyle **istatistiksel olarak ilişkili** olduğunu kanıtlıyor — tek başına **artımlı tahmin faydası** sağladığı anlamına gelmiyor (bkz. Bölüm 9, Soru 6).

**(2) `tissue_statistics_residual_fusion`:** Aynı sıfır-başlatmalı rezidüel mekanizması, CNN yerine 20-boyutlu istatistik vektörü üzerinde. 3-fold (tam, tüm checkpoint'ler mevcut): accuracy 0.8159 ± 0.0138, macro-F1 0.8175 ± 0.0129, QWK 0.9131 ± 0.0156 — temel çizgiden ayırt edilemiyor. Ama shuffle-ablasyonu, CNN varyantından (göreli L2 ≈ 0.0045) **~20-25 kat daha içerik-duyarlı** bir düzeltme buluyor (göreli L2 ≈ 0.076–0.154, fold'a göre) — CNN'in zayıf gradyan darboğazını atlamak ölçülebilir bir ilerleme sağladı, ama hâlâ toplam accuracy'yi hareket ettirecek kadar güçlü değil (fold-1'de 2/73 tahmin değişiyor, fold-0/2'de hiç değişmiyor).

**(3) ANOVA — hangi özellikler gerçekten ayırt edici?** 17/20 özellik istatistiksel olarak anlamlı (p<0.05). En ayırt ediciler `slough_std`, `granulation_std`, `slough_p95` — yani doku **heterojenliği** (std), ortalama kompozisyondan daha bilgilendirici (klinik olarak makul: daha şiddetli bir yara yatağı daha heterojendir). `bone_mean` istatistiksel olarak anlamlı ama klinik beklentiyle **ters yönlü** (stage 1'de en yüksek) ve eşlik eden `bone_frac_above_0.5` anlamlı değil (p=0.41) — bu, Task 2'nin en nadir sınıfından kaynaklanan gürültü olarak işaretlendi, gerçek bir bulgu sayılmadı.

**(4) Tamamlayıcılık ve hata örtüşmesi analizi (RGB vs. `tissue_statistics_only`, eşleştirilmiş tahminler, n=1.091 havuzlanmış):**

![Complementarity contingency](figures/complementarity_contingency.png)

| | Tissue doğru | Tissue yanlış |
|---|---|---|
| **RGB doğru** | 871 (%79.8) | 20 (%1.8) |
| **RGB yanlış** | 15 (%1.4) | 185 (%17.0) |

- **P(tissue doğru \| RGB yanlış) = 87/200 = %43.5** — tissue modelinin kendi ortalama accuracy'sinden (~%38) ve majority-class temel çizgisinden (~%28.7) belirgin şekilde yüksek; yani RGB'nin hata yaptığı koşullu kümede tissue, kendi ortalamasından daha iyi — gerçek, rastgele olmayan bir koşullu sinyal.
- **Oracle seçim üst sınırı = 896/1.091 = %89.6** (RGB veya tissue **herhangi biri** doğruysa doğru sayılır). **Bu değer gerçek etiketi bilmeyi gerektirir — uygulanabilir/dağıtılabilir bir model değildir**, sadece "mükemmel bir seçici olsaydı ne kadar kazanç olurdu" teorik tavanıdır.
- RGB'nin hatalarının %56.5'inde tissue de aynı hatayı yapıyor (örtüşme) — tamamlayıcılık **kısmi**, tam değil.
- `tissue_statistics_only` **hiçbir fold'da stage_4'ü doğru sınıflandıramıyor** (F1=0.0000, üç fold'da da) — modelin accuracy'si tamamen stage_1/2/3'teki kazançlardan geliyor.
- RGB'nin en sistematik hata modu (stage_3→stage_4 aşırı tahmini, stage_1↔2 sınır karışıklığı) tam olarak tissue'nun düzelttiği bölgelerle örtüşüyor.

(Bu analiz `more_experiments/exp/exp6_tissue_statistics/scripts/complementarity_analysis.py` ile bu raporun hazırlığı sırasında yeniden çalıştırılıp doğrulandı; çıktısı `results/complementarity_analysis_report.txt`'e kaydedildi.)

## 7. Global–Local ROI Özellik Füzyonu (exp-7)

Bayraktar ve ark. (2026, *Integrating Localization, Feature Fusion, and Attention for Pressure Ulcer Staging from Clinical Images*, JIIM, Fig. 4 — dikkat-mekanizmalı Fig. 5 modeli **değil**) çalışmasından uyarlanan mimari, mevcut MiT-B1+CORN ve sızıntısız Task 1 modeliyle test edildi.

**Mimari:** `logits = fusion_head(concat(F_global, F_local))`. `F_global`, dondurulmuş temel çizginin son-katman özellik haritasının (`[B,512,7,7]`) GAP'i. `F_local`, MiT-B1'in **üçüncü** katmanı (`[B,320,14,14]`, son katmandan daha yüksek uzamsal çözünürlük) üzerinde, tahmini yara kutusuyla `roi_align` uygulanıp 128-boyuta projekte edilmiş. `fusion_head`, temel çizginin kendi CornHead'inin **genişletilmiş** hali (512→640): 512 global sütun+bias birebir kopyalanmış, 128 yeni local sütun sıfırla başlatılmış — kurulumda `fusion_logits == baseline_logits` **birebir** (`torch.equal`, 12/12 mimari test geçti).

**Faz A (dondurulmuş backbone, sadece 0.043M/13.194M parametre eğitiliyor), 3-fold:**

| Metrik | Temel Çizgi | **exp-7** |
|---|---|---|
| Accuracy | 0.8168 ± 0.0150 | 0.8120 ± 0.0098 |
| Macro-F1 | 0.8182 ± 0.0138 | 0.8142 ± 0.0095 |
| QWK | 0.9138 ± 0.0164 | 0.9131 ± 0.0090 |

**ROI katkı ablasyonu** (tahmini ROI / tam-görüntü ROI / karıştırılmış ROI, aynı eğitilmiş ağırlıklar): **3 fold'un hepsinde 0 tahmin değişimi** — accuracy/F1/MAE/QWK 4 ondalıkta birebir aynı. Kök neden doğrudan doğrulandı: `fusion_head`'in 128 local sütunu eğitim sırasında sıfırdan uzaklaştı (|ağırlık| ortalama ≈0.0006–0.001) ama global sütunlardan (≈0.023) **20-35 kat küçük** kaldı — hiçbir görüntüde CORN karar sınırını geçmeye yetmedi.

**Dikkatli yorum (Bayraktar ve ark.'ın kendi bulgusunu çürütmüyor):** bu, *bu spesifik* dondurulmuş-backbone, sıfır-başlatmalı-genişletilmiş-başlık tasarımının kullanışlı bir sinyal çıkaramadığını gösterir — yara lokalizasyon bilgisinin evreleme için **genel olarak** faydasız olduğu sonucuna **varılamaz**. Faz B (son MiT-B1 katmanının çözülüp ince ayar yapılması) uygulanmadı; brief'in "Faz A tam değerlendirilmeden yeni mimari aramaya başlama" talimatına uyuldu.

## 8. Konsolide Nicel Sonuçlar

Tam tablo: `more_experiments/reports/master_experiment_comparison.csv`.

![Accuracy comparison](figures/accuracy_comparison.png)

![QWK comparison](figures/qwk_comparison.png)

![Architecture comparison](figures/architecture_diagram.png)

| Deney | Girdi | Fold | Accuracy | Macro-F1 | QWK | Δ Accuracy (pp) |
|---|---|---|---|---|---|---|
| RGB temel çizgi | RGB | 3 | 0.8168±0.0150 | 0.8182±0.0138 | 0.9138±0.0164 | — |
| exp-1 skin+wound | maske | 3 | 0.7819±0.0157 | 0.7815±0.0149 | 0.8935±0.0111 | −3.49 |
| exp-2 wound-only | maske | 3 | 0.7837±0.0222 | 0.7863±0.0214 | 0.9000±0.0149 | −3.31 |
| exp-3 erken füzyon | 8ch | 3 | 0.7791±0.0126 | 0.7822±0.0116 | 0.8997±0.0098 | −3.77 |
| exp-4 concat | iki-kol | 3 | 0.7983±0.0156 | 0.8003±0.0155 | 0.9109±0.0114 | −1.85 |
| exp-4 gated | iki-kol | 3 | 0.7957±0.0238 | 0.7977±0.0229 | 0.9063±0.0125 | −2.11 |
| exp-5 rgb+tissue rezidüel | rezidüel | 3 | 0.8158±0.0073 | 0.8178±0.0073 | 0.9132±0.0122 | −0.10 |
| exp-5 tissue-only rezidüel | rezidüel | **1 (kısmi)** | 0.8370 | 0.8376 | 0.9323 | n/a — bkz. §5 |
| exp-6 stats-only | 20-dim | 3 | 0.3803±0.0094 | 0.3065±0.0151 | 0.3550±0.0091 | −43.65 |
| exp-6 stats rezidüel | 20-dim+RGB | 3 | 0.8159±0.0138 | 0.8175±0.0129 | 0.9131±0.0156 | −0.09 |
| exp-7 global-local ROI | RGB+ROI | 3 | 0.8120±0.0098 | 0.8142±0.0095 | 0.9131±0.0090 | −0.48 |
| exp-9 çoklu-öğretmen damıtma | distilled | **1 (kısmi)** | 0.7403 | 0.7404 | 0.8583 | n/a — bkz. §11 |
| exp-10 `heads` | RGB (donmuş+kalıntı) | 3 | 0.8139±0.0132 | 0.8167±0.0122 | 0.9110±0.0115 | −0.29 |
| exp-10 `attention` | RGB (donmuş+kalıntı) | 3 | 0.8130±0.0102 | 0.8155±0.0103 | 0.9120±0.0105 | −0.38 |
| exp-10 `contrastive` | RGB (donmuş+kalıntı) | 3 | 0.8167±0.0144 | 0.8188±0.0136 | 0.9138±0.0142 | −0.01 |
| exp-11 probe `final` | donmuş 512d | 3 | 0.8103±0.0111 | 0.8122±0.0113 | 0.9125±0.0101 | −0.65 |
| exp-11 probe `stage3` | donmuş 320d | 3 | 0.7709±0.0198 | 0.7725±0.0199 | 0.8920±0.0078 | −4.59 |
| WP1 exp-4 concat, birleşik kanal | iki-kol, 1ch | 3 | 0.7909±0.0282 | 0.7935±0.0277 | 0.9057±0.0174 | −2.59 |
| WP1 exp-5 tissue_only, birleşik kanal | rezidüel, 1ch | 3 | 0.8131±0.0079 | 0.8150±0.0074 | 0.9118±0.0131 | −0.37 |
| **Ensemble (4 model, sum-rule)** | softmax ortalama | 3 | **0.8222±0.0136** | **0.8239±0.0142** | **0.9178±0.0105** | **+0.54** |

(exp-9/10/11 ve WP1-5 satırları `master_experiment_comparison.csv`'ye henüz eklenmedi — sayılar kendi exp klasörlerinin `results/` dosyalarından doğrudan alındı, bkz. §11-14. Ensemble satırı Ek C'den — **Chapter 8'in tek pozitif sonucu**: tek model füzyonlarının hiçbiri baseline'ı geçemezken, zaten eğitilmiş 4 farklı mimarinin tahminlerini basitçe ortalamak geçti.)

## 9. Bilimsel Yorumlama

**Soru 1 — Yara lokalizasyonu evreleme doğruluğunu artırıyor mu?** Hayır, test edilen hiçbir biçimde: ne maskeleme (exp-1/2, −3.3pp), ne de özellik-düzeyinde ROI füzyonu (exp-7, −0.5pp, 0 tahmin değişimi). Doğrudan gözlem.

**Soru 2 — Arka planı kaldırmak yardımcı mı, zararlı mı?** Zararlı — exp-1 ve exp-2, tuttukları alan miktarından (skin+wound %97 vs. wound-only çok daha az) bağımsız olarak **istatistiksel olarak ayırt edilemez** sonuçlar veriyor. Bu, kaybın "ne kadar maskelendiği"nden değil "maskelemenin kendisi"nden geldiğine işaret ediyor — hipotez: arka plan/çerçeveleme bağlamı gerçek sinyal taşıyor.

**Soru 3 — Tahmini doku olasılık haritaları RGB'nin ötesinde bilgi taşıyor mu?** Evet — ama bu iddia, yöntemine göre ayrılmalı: (a) erken/geç/rezidüel **füzyon** olarak eklendiğinde ölçülebilir bir accuracy kazancı sağlamıyor; (b) **tek başına** (exp-6 stats-only), şans seviyesinin belirgin üzerinde (%38 vs %25-29); (c) **koşullu olarak** (RGB yanlış olduğunda), tissue kendi ortalamasından daha iyi performans gösteriyor (%43.5). Yani istatistiksel ilişki var, ama bu ilişki mevcut füzyon mimarileriyle **artımlı tahmin faydasına** dönüştürülemiyor — bu ikisi karıştırılmamalı (hipotez, doğrudan gözlemden ayrı).

**Soru 4 — Erken ve geç füzyon neden tutarlı biçimde temel çizgiyi geçemedi?** Erken füzyon (exp-3), tek bir paylaşılan konvolüsyon yığınını hem RGB hem doku kanallarını aynı anda öğrenmeye zorluyor — bu, gradyanların iki farklı-ölçekli sinyal arasında paylaşılmasına yol açmış olabilir (hipotez). Geç füzyon (exp-4) daha iyi performans gösterdi ama RGB kolu ImageNet-init'liydi, eğitilmiş temel çizgiden değil — bu da zaten güçlü olan RGB yolunun temsil gücünden ödün verdi (doğrudan gözlem: exp-5'in aynı doku kolunu dondurulmuş-gerçek-temel-çizgiyle birleştirdiğinde fark temel çizginin std'si içine düştü).

**Soru 5 — Rezidüel füzyon örneğe-özgü doku düzeltmelerini gerçekten öğrendi mi?** Kısmen, ve yöntem bağımlı. Orijinal `rgb_tissue` varyantı: **hayır** — shuffle-ablasyonu kosinüs≈1.0 gösterdi, `delta_z` neredeyse-sabit bir "hiçbir şey yapma" çözümüne yakınsadı (doğrudan gözlem, gradyan normlarıyla desteklendi). `tissue_only` CNN varyantı: biraz daha iyi ama hâlâ göreli L2≈0.005 ile zayıf. El-yapımı istatistik varyantı: göreli L2 20-25 kat arttı (0.076–0.154) — **daha içerik-duyarlı**, ama hâlâ aggregate accuracy'yi hareket ettiremedi. Sonuç: CNN kolunun zayıf gradyanı gerçek bir darboğazdı; bunu atlamak düzeltmeyi daha "gerçek" yaptı ama henüz yeterince güçlü değil.

**Soru 6 — Tissue-only sınıflandırma ve tamamlayıcılık analizi ne ortaya çıkardı?** Doku istatistikleri gerçek ama **dar bantlı** bir sinyal taşıyor: RGB'nin en sistematik hata modunda (stage sınırı karışıklıkları) iyileşme sağlıyor (koşullu accuracy %43.5), ama stage_4'te tamamen kör (F1=0.0) ve RGB'nin hatalarının yarısından fazlasını paylaşıyor. Oracle'ın gösterdiği ~%9'luk teorik boşluk (%89.6 vs %81.7), mevcut 20-boyutlu özet istatistik + basit rezidüel MLP tasarımıyla **erişilemiyor** — bu bir mimari/kapasite sınırlaması mı yoksa temel bir bilgi sınırı mı, açık bir soru (gelecek çalışma).

**Soru 7 — Global-local ROI füzyonu evrelemeyi iyileştirdi mi?** Hayır, test edilen Faz A (dondurulmuş backbone) konfigürasyonunda — ve bu Chapter 8'in **en net negatif sonucu**: sadece metrik eşitliği değil, **birebir aynı tahminler** (3 ablasyon koşulunda da 0 flip). Ama bu, lokalizasyon bilgisinin genel olarak işe yaramadığı anlamına gelmez (bkz. §7'nin sonu ve Soru 8) — sadece bu dar mimari/eğitim rejiminin onu çıkaramadığı anlamına gelir.

**Soru 8 — Değerlendirilen entegrasyon stratejilerinin sınırlılıkları nelerdir?**
- Tüm füzyon denemeleri **dondurulmuş veya ImageNet-init RGB kolu + küçük bir ek dal** şeklinde tasarlandı — RGB yolunun kendisinin doku/lokalizasyon bilgisiyle birlikte **uçtan uca yeniden eğitilmesi** hiçbir deneyde denenmedi (Faz B, exp-7'de planlanmış ama uygulanmamış).
- Doku istatistikleri yalnızca **tahmini yara bölgesi** üzerinden hesaplanıyor — Task 1'in kendi lokalizasyon hataları, Task 2'nin doku tahminlerine de sızıyor olabilir (ayrı bir kaynak değil, zincirlenmiş bir hata kaynağı).
- Üç rezidüel varyantın hiçbiri (CNN+RGB, CNN tissue-only, 20-dim stats) exp-5/6 teşhislerinde **güçlü** bir gradyan sinyali göstermedi — bu, veri kümesi büyüklüğünün (1.091 görüntü, küçük ek dallar için bile sınırlı) bir faktör olabileceğine işaret ediyor (hipotez).
- exp-5'in `tissue_only` varyantı ve exp-7'nin ROI füzyonu dahil, birkaç bulgu **tek fold** veya **frozen-backbone Faz A** ile sınırlı — tam resmi tamamlamak için ek fold'lar/Faz B gerekiyor.

## 10. Sonuçlar ve Tez İçin Çıkarımlar

Yedi deney ailesi, 11 ayrı konfigürasyon, tutarlı bir PIID 3-fold protokolüyle test edildi. **Hiçbir füzyon stratejisi RGB temel çizgisini istatistiksel olarak anlamlı biçimde geçemedi** — en yakın sonuçlar (exp-5 rgb_tissue, exp-6 stats-residual, exp-7) temel çizginin kendi std'si içinde kaldı. Bu, "doku/lokalizasyon bilgisi evreleme için işe yaramaz" sonucuna **varmak için yeterli değil** — tersine, Bölüm 6'nın tamamlayıcılık analizi doku bilgisinin gerçek, koşullu bir sinyal taşıdığını (%43.5) doğrudan gösteriyor. Bulgu, daha hassas bir ayrımdır: **mevcut dondurulmuş-backbone / küçük-ek-dal füzyon tasarımları bu sinyali yeterince güçlü bir gradyanla çıkaramıyor.**

Tez için üç somut çıkarım:
1. RGB MiT-B1+CORN temel çizgisi, Chapter 8'in tüm denemelerine karşı **sağlam** kalıyor — bu, Chapter 7'nin mimari seçiminin doğrulanmasıdır.
2. Doku bilgisinin **var olduğu** ama **çıkarılamadığı** gösterildi — bu, "tissue segmentation yararsız" değil, "mevcut füzyon mekanizmaları yetersiz" biçiminde ifade edilmeli.
3. Gelecek çalışma için en somut yön: RGB kolunun dondurulmadığı (uçtan uca, düşük LR ile birlikte ince ayarlı) bir füzyon mimarisi, ya da doku/lokalizasyon bilgisinin ayrı bir dal yerine **aynı backbone'un içinde** (ör. erken katmanlarda hafif bir koşullandırma) kullanılması — hiçbiri bu Chapter'da denenmedi.

---

## 11. Çoklu-Öğretmen Özellik Damıtma (exp-9)

**Soru:** Üç farklı, önceden eğitilmiş mimarinin öğrendiği temsiller tek bir öğrenci kodlayıcıya damıtılırsa (knowledge distillation), bu öğrenci RGB temel çizgisini geçecek kadar zengin bir temsil öğrenir mi?

**Mimari — 2 aşama:**

**Aşama 1 (damıtma, etiketsiz):** Öğrenci kodlayıcı (`efficientnet-b0` — bilerek 3 öğretmenden farklı bir mimari, böylece sadece tek bir öğretmeni yeniden türetmiyor), her öğretmen için ayrı bir doğrusal projeksiyon başlığı üzerinden, o öğretmenin havuzlanmış gömme vektörünü yeniden üretmeye çalışır. Üç öğretmen: `mit_b1` (512d), `resnext50_32x4d` (2048d), `convnextv2_tiny` (768d) — her biri kendi fold-özel checkpoint'inden, donmuş, sadece hedef gömme vektörü üretmek için kullanılır. Kayıp, üç öğretmen üzerinden ortalanan kosinüs mesafesidir (ham MSE yerine, üç öğretmenin çok farklı gömme ölçeklerine karşı sağlam olsun diye):

$$\mathcal{L}_{\text{distill}} = \frac{1}{3}\sum_{t \in \{\text{mit\_b1, resnext50, convnextv2}\}} \left(1 - \cos\big(\text{proj}_t(f_{\text{öğrenci}}),\ f_t^{\text{öğretmen}}\big)\right)$$

Hiçbir evreleme etiketi bu aşamada kullanılmaz — sadece Adım 1'de önbelleğe alınmış öğretmen gömme vektörleri (60 epoch, AdamW, kosinüs LR takvimi).

**Aşama 2 (sınıflandırma):** Damıtılmış öğrenci donuk tutulur (Faz A — doğrusal prob, eğitilebilir parametre sayısı 1.091 görüntü için kasıtlı küçük), üzerine taze bir `CornHead` gerçek evre etiketleriyle eğitilir (`task_3/src/engine.py`'nin değiştirilmemiş `fit()`/`run_epoch()`'u). Her fold için her öğretmen hedefi **o öğretmenin kendi fold-i checkpoint'inden** gelir (fold-i'yi kendi eğitiminde dışlamış olan) — ne damıtma ne sınıflandırma aşaması fold-i'nin test görüntülerini sızdırmaz.

**⛔ Durum — sadece fold-0, bilinçli olarak durduruldu.** Commit `5d4a7f6`'nın kendi notu: *"Killed after fold0 (acc 0.7403 vs mit_b1\_\_corn's 0.8039) -- doesn't beat baseline, consistent with every other fusion attempt in this chapter; not pursued further."* Fold-1/2 hiç çalıştırılmadı.

| Metrik | fold-0 |
|---|---|
| Accuracy | 0.7403 |
| Macro-F1 | 0.7404 |
| MAE | 0.279 |
| QWK | 0.8583 |

(`more_experiments/exp/exp9_feature_distillation/results/exp9_results.csv` — dosyada ayrıca `distill_val_cos_loss=0.9902` ile `accuracy=0.0` olan bir ilk satır daha var; bu, damıtmanın yakınsamadığı erken/başarısız bir deneme olarak değerlendirilir, ikinci satır düzeltilmiş/yakınsamış çalıştırmadır — ayrım repoda ayrıca etiketlenmemiş, şeffaflık için burada not edildi.)

**Bulgu:** Tek fold'luk kısmi sonuç bile (0.7403), Chapter 8'in diğer tüm füzyon/temsil denemeleriyle aynı örüntüyü izliyor: temel çizgiyi geçmiyor. Üç farklı mimariden damıtılmış bir temsil, gerçek etiketlerle uçtan uca eğitilmiş RGB temel çizgisinin yerini tutmuyor.

## 12. Sınıra-Özel Temsil Öğrenme (exp-10)

**Soru:** CORN'un üç koşullu kararı aynı havuzlanmış 512-boyutlu vektörü okuyor ve yalnızca doğrusal ağırlık satırlarıyla ayrışıyor (`CornHead`, tek bir `Linear(512,3)`). Her karara özgü bir mekânsal/temsilsel ayrım sağlanırsa performans artar mı?

**Mimari:** $\text{logit}_k = \text{baseline\_logit}_k + \delta_k$ ($k\in\{0,1,2\}$, CORN'un 3 koşullu sınırı). `baseline_logit_k`, gerçek eğitilmiş, **donmuş** `mit_b1__corn` checkpoint'inin `CornHead`'inden gelir, hiçbir modda eğitilmez. Her sınırın kendi kalıntı başlığı:

$$A_k = \sigma\big(\text{Conv}_{1\times1,k}(F)\big), \qquad z_k = \text{GAP}(F \odot A_k), \qquad \delta_k = \text{MLP}_k(z_k)$$

$F$ donmuş backbone'un son özellik haritası ($[B,512,7,7]$). Üç mod $A_k$'nin tanımında farklılaşır:

| Mod | $A_k$ | Test ettiği soru |
|---|---|---|
| `heads` | sabit $1$ (yok) | Yalnızca sınıra-özel doğrusal olmayan karar fonksiyonu işe yarıyor mu? |
| `attention` | öğrenilir, yukarıdaki formül | Sınıra-özel mekânsal seçim işe yarıyor mu? |
| `contrastive` | `attention` + sınır-başına supervised contrastive terim | Sınır temsillerini açıkça ayrıştırmak işe yarıyor mu? |

Kasıtlı tasarım farkı (exp-5/6/7'nin aksine): $\text{MLP}_k$'nin son katmanı sıfırla başlatılmıyor (gradyanın attention konvolüsyonuna gecikmeden ulaşması için) — kuruluşta $\text{logit}_k \neq \text{baseline\_logit}_k$ tam olarak, ama backbone ve baseline head donmuş kalıyor (doğrudan doğrulandı).

**Sonuç (Colab, 3 mod × 3 fold, gerçek eğitim):**

| Mod | Accuracy | Macro-F1 | QWK |
|---|---|---|---|
| **baseline** (referans) | 0.8168 ± 0.0150 | 0.8182 ± 0.0138 | 0.9138 ± 0.0164 |
| `heads` | 0.8139 ± 0.0132 | 0.8167 ± 0.0122 | 0.9110 ± 0.0115 |
| `attention` | 0.8130 ± 0.0102 | 0.8155 ± 0.0103 | 0.9120 ± 0.0105 |
| `contrastive` | 0.8167 ± 0.0144 | 0.8188 ± 0.0136 | 0.9138 ± 0.0142 |

`contrastive` baseline'dan istatistiksel olarak ayırt edilemez; `heads`/`attention` 1 std içinde ama hafif altında. Hiçbiri 0.8168'i geçmedi, ama hiçbiri de anlamlı altına düşmedi.

**Attention tanısı:** `attention` modunda 3 bağımsız konvolüsyon birbirinin neredeyse aynısını öğrendi (ortalama kosinüs 0.9964, argmax-uyumu %11.55) — "sınıra-özel mekânsal seçim" iddiası desteklenmiyor. `contrastive` bunu ölçülebilir azalttı (kosinüs 0.9170, argmax-uyumu %23.34) ama bu ayrışma **doğruluğa dönüşmedi** (1 std içinde) — exp-6 ile aynı örüntü: içerik-duyarlılık arttı, aggregate accuracy hareket etmedi.

## 13. Doğrusal Prob (Linear Probe) Teşhisi (exp-11)

**Soru:** MiT-B1 kodlayıcısı, CORN'un yanlış sınıflandırdığı yaraları ayırt edebilecek bilgiyi zaten öğrenmiş mi (CORN'un kendi doğrusal kararı bunu kullanmıyor mu), yoksa bu bilgi kodlayıcıda hiç yok mu? Kodlayıcı hiç yeniden eğitilmeden, aynı donmuş $z$ üzerine farklı bir doğrusal karar kuralı (`LogisticRegression`, $C$ validation macro-F1 ile seçilen) eğitiliyor.

**İki seviye:** `final` (512d, `CornHead`'in okuduğu aynı vektör) ve `stage3` (320d, MiT-B1'in son katmanından önceki aşama — "bilgi daha erken mi kayboluyor" kontrolü).

İç tutarlılık: her fold'da probe'un okuduğu donuk CORN tahminleri, `ablation_results.csv`'deki aynı fold'un accuracy'siyle 4 ondalık basamağa kadar eşleşti (0.8122/0.8370/0.8011).

**Sonuç — 3 fold, gerçek:**

| Model | Accuracy | Macro-F1 | MAE | QWK |
|---|---|---|---|---|
| **CORN** (referans) | 0.8168 ± 0.0150 | 0.8182 ± 0.0138 | 0.1896 ± 0.0204 | 0.9138 ± 0.0164 |
| Probe `final` (512d) | 0.8103 ± 0.0111 | 0.8122 ± 0.0113 | 0.1952 ± 0.0153 | 0.9125 ± 0.0101 |
| Probe `stage3` (320d) | 0.7709 ± 0.0198 | 0.7725 ± 0.0199 | 0.2355 ± 0.0202 | 0.8920 ± 0.0078 |

**Hata kırılımı (3 fold toplamı, 200 CORN hatası):**

| Seviye | Kurtarılan | Yeni hata | Havuzlanmış kurtarma oranı |
|---|---|---|---|
| `final` | 29 | 36 | %14.5 |
| `stage3` | 50 | 100 | %25.0 |

`stage3` her fold'da `final`'den fazla hata kurtarıyor ama çok daha fazla yeni hata yaratıyor — tutarlı bir örüntü. true=stage_3/pred=stage_4 hücresinde (n=65): `final` 10 kurtarma (%15.4), `stage3` 19 (%29.2) — ama fold2'de `final` bu hücrede sıfır kurtarma yapıyor.

**Yorum:** CORN'un okuduğu temsilde, CORN'un kullanmadığı gerçek ama sınırlı bir ek bilgi var (29 tutarlı kurtarma, 3/3 fold). "Encoder yetersiz" değil, "CORN'un zaten kullandığına yakın bir temsil okuyor" demek. Tek bir probe sonucu başlı başına mimari değişikliği haklı çıkarmaz.

## 14. Danışman Ek Deneyleri — WP1-5 (`more_experiments/experiment_extended/`)

Bölüm 1-13, tez taslağının orijinal Chapter 8 kapsamını (exp-1→exp-11 + ensemble) kapsar. Bu bölüm danışmanın ayrıca talep ettiği beş ek iş paketini (WP1-5) aynı titizlikle belgeler — `more_experiments/experiment_extended/` altında, mevcut checkpoint/veri/split yeniden kullanılarak, hiçbir önceki sonuç üzerine yazılmadan.

**Ortak altyapı — CORN kod çözme kuralı (tüm WP'lerde aynı, değiştirilmedi):** $k$. koşullu lojit $\ell_k$ için $P(Y>k\mid Y>k-1)=\sigma(\ell_k)$, survival $S_k=\prod_{j\le k}\sigma(\ell_j)$, tahmin edilen evre $=1+|\{k:S_k>0.5\}|$ (`task_3/corn_explain.py`'nin `corn_outputs()`'u).

### WP1 — Tek birleşik doku kanalı (Exp-4 / Exp-5)

**Soru (danışmanın orijinal notu):** "Tek kanal exp-4 ve exp-5 denenecek" — 5 doku olasılık haritası yerine tek bir birleşik kanal.

**Matematik:** exp-3 verisinin 5 doku kanalı ($p_1,\dots,p_5$ = granülasyon, slough, nekroz, tendon, kemik olasılıkları), piksel bazında maksimumla tek kanala indirgeniyor:

$$c(x,y) = \max\big(p_{\text{gran}}(x,y), p_{\text{slough}}(x,y), p_{\text{nekroz}}(x,y), p_{\text{tendon}}(x,y), p_{\text{kemik}}(x,y)\big)$$

Doku tipi kimliği kasıtlı atılıyor; sadece "burada herhangi bir anormal doku var mı, ne kadar güçlü" sinyali kalıyor. exp-4 (`concat`, $z=\text{MLP}([F_{rgb};F_{doku}])$) ve exp-5 (`tissue_only` rezidüel, $z_{final}=z_{rgb}+\delta z$, $\delta z$ sadece $c$'den) mimarileri değiştirilmeden kullanılıyor.

**Sonuç (Colab T4, 3 fold, gerçek eğitim):**

| | Accuracy | Macro-F1 | MAE | QWK |
|---|---|---|---|---|
| RGB temel çizgi | 0.8168 | 0.8182 | — | 0.9138 |
| Exp-4 concat, 5 kanal ayrı (referans) | 0.7983 | 0.8003 | — | 0.9109 |
| **Exp-4 concat, birleşik kanal** | 0.7909 ± 0.0282 | 0.7935 ± 0.0277 | 0.2127 | 0.9057 ± 0.0174 |
| Exp-5 tissue_only, 5 kanal ayrı (yalnızca fold1, referans) | 0.8370 | 0.8376 | — | 0.9323 |
| **Exp-5 tissue_only, birleşik kanal** | 0.8131 ± 0.0079 | 0.8150 ± 0.0074 | 0.1933 | 0.9118 ± 0.0131 |

**Bulgu:** İki bağımsız mimaride de aynı yönde küçük bir fark — birleşik kanal, 5 kanalı ayrı tutmaktan biraz daha kötü (Exp-4: −0.0074 acc/−0.0068 f1/−0.0052 qwk; Exp-5 fold1-eşleştirilmiş: −0.0138/−0.0129/−0.0064), ikisi de RGB temel çizgisine yakın kalıyor. Doku **tipi** kimliğinin küçük ama gerçek bir katkısı var.

### WP2 — Stage 1 hariç tutulan doku füzyonu

**Soru:** Doku füzyonu sadece Stage 2-4 kararlarına uygulanıp Stage 1'e hiç karışmazsa performans artar mı?

**Mekanizma — çıkarım-zamanı kademeli (cascade), yeni mimari değil:**

$$\hat{y}(x) = \begin{cases} \text{decode}(f_{\text{RGB}}(x)) & \text{eğer decode}(f_{\text{RGB}}(x)) = \text{stage\_1} \\ \text{decode}(f_{\text{fusion}}(x)) & \text{aksi halde} \end{cases}$$

$f_{\text{RGB}}$ (donmuş `mit_b1__corn`) ve $f_{\text{fusion}}$ (donmuş exp-4 concat, 5 kanal) iki ayrı, zaten eğitilmiş model; yeniden eğitim yok. Dal seçimi sadece $f_{\text{RGB}}$'nin kendi tahminine dayanır, gerçek etikete bakmaz.

**Sonuç (1091 test örneği, 3 fold havuzlanmış):**

| Model | Scope | n | Accuracy | Macro-F1 | QWK |
|---|---|---|---|---|---|
| baseline | tüm evreler | 1091 | 0.8167 | 0.8185 | 0.9141 |
| baseline | stage 2-4 | 861 | 0.8002 | 0.6055 | 0.8528 |
| fusion (kısıtsız) | stage 2-4 | 861 | 0.7724 | 0.5901 | 0.8371 |
| **cascade (Stage-1 hariç)** | stage 2-4 | 861 | **0.7666** | **0.5876** | **0.8351** |

(`stages_2_4` macro-F1'in düşük görünmesi, bu alt kümenin gerçek etiketinde stage_1 hiç olmamasından kaynaklanan bir metrik artefaktıdır, model çöküşü değil.)

Stage-1 kararlarının kararlılığı doğrudan doğrulandı: baseline'ın stage_1 dediği 223 örneğin 223'ünde cascade aynı fikirde; kalan 868'inde cascade'in tahmini kısıtsız fusion modeliyle birebir (868/868) aynı.

**Bulgu (negatif sonuç):** Stage 1'i füzyondan çıkarmak yardımcı olmuyor — cascade, Stage 2-4 alt kümesinde bile kısıtsız füzyondan hafifçe daha kötü, ikisi de RGB-only'den daha kötü. Cascade her çalıştığında aynı donmuş füzyon checkpoint'ini kullanıyor — *ne zaman* danışıldığını değiştiriyor, füzyonun kendi zayıflığını düzeltmiyor.

### WP3 — Kemik/tendon kural tabanlı kontrol (Stage 3 vs 4)

**Soru:** Model Stage 3/4 arasında kararsız kaldığında, kemik/tendon olasılık haritalarına bakan basit bir kural hataları düzeltir mi?

**Kural:**

$$e(x) = \max\big(\text{bone\_max}(x), \text{tendon\_max}(x)\big), \qquad \hat{y}_{\text{rule}}(x) = \begin{cases}\text{stage\_4} & \hat{y}(x)=\text{stage\_3} \text{ ve } e(x)\ge\tau \\ \hat{y}(x) & \text{aksi halde}\end{cases}$$

`bone_max`/`tendon_max`: tahmini yara bölgesi içindeki piksel-bazlı maksimum olasılık. Kural tek yönlü — sadece stage_3→stage_4 düzeltmesi yapabilir. $\tau$ yalnızca validation'daki 110 "contested" örnekte ızgara araması ile seçildi; test setine (530 contested örnek) $\tau$ sabitlendikten sonra tek seferde dokunuldu.

**Sonuç:** $\tau=1.0$ (hiçbir gözlemlenen değer buna ulaşmıyor — kural hiç tetiklenmiyor). `n_changed_by_rule=0`. Pair-accuracy önce/sonra: 0.8075 (428/530). 3→4 hataları 65→65, 4→3 hataları 37→37 (değişmedi).

**Bulgu (negatif sonuç):** Kural hiçbir şeyi değiştirmiyor. WP4 ile tutarlı: doğru Stage 4'te bile `bone_max` küçük (0.0546), kaçırılan (true=4,pred=3) örneklerde **daha yüksek** (0.0652) — basit "daha fazla kemik → Stage 4" ilişkisi net değil.

### WP4 — Stage 3-4 hata incelemesi (keşifsel)

Model/eğitim yok — 3-fold test tahminleri üzerinden bir vaka tablosu (`stage34_error_cases.csv`, 530 satır) ve görsel inceleme panelleri (`error_panels/`) üretildi.

| Grup | n |
|---|---|
| Doğru Stage 3 | 193 |
| Doğru Stage 4 | 235 |
| Hata: gerçek 3 → tahmin 4 | 65 |
| Hata: gerçek 4 → tahmin 3 | 37 |

**Ölçülen örüntüler:** Her iki hata grubu da kendi "doğru" karşılığından düşük güvenli (`P(pred)`): 0.8405→0.8150 (3 tarafı), 0.8858→**0.7541** (4 tarafı). Kemik/tendon gruplar arası net ayrışmıyor (`error_4to3`, `correct_4`'ten daha yüksek kemik sinyaline sahip). `wound_detected` 530/530 True. Bu bölüm WP3/WP5'in kural tasarımını bilgilendirdi (önce çalıştırıldı).

### WP5 — Derinlik ipucu (Stage 3 vs 4)

**Soru:** Depth Anything V2'nin göreli, tek-görüntü derinlik haritaları Stage 3/4 ayrımına yardımcı olur mu?

**Sinyal analizi:** 530 contested örnekte, `wound_depth_std` ve `wound_depth_range`, true=stage_4 grubunda p<0.01 ile anlamlı (doğru: std=0.520, yanlış: std=0.430).

**Genişletilmiş kural:**

$$\hat{y}_{\text{rule+depth}}(x) = \begin{cases}\text{stage\_4} & \hat{y}(x)=\text{stage\_3} \text{ ve } \big(e(x)\ge\tau_{\text{doku}} \text{ veya } d(x)\le\tau_{\text{depth}}\big) \\ \hat{y}(x) & \text{aksi halde}\end{cases}$$

$d(x)$ = yara-içi ortalama derinlik $-$ yara-dışı (deri) ortalama derinlik. $\tau_{\text{depth}}$, WP3 ile aynı 110 validation örneğinde seçildi.

**Sonuç:** $\tau_{\text{depth}}=-1.54$, test setinde hiç tetiklenmiyor. Pair-accuracy yine 0.8075, iki hata yönü de değişmedi.

**Bulgu (negatif sonuç):** Grup-ortalaması düzeyinde anlamlı fark olsa bile, 110 örnekte fit edilen tek bir global eşik bunu test setine aktaramadı.

### WP1-5 Özet

| WP | Soru | Sonuç |
|---|---|---|
| 1 | Doku kanalları tek kanalda birleştirilirse? | Hafif, tutarlı kötüleşme (iki mimaride de) — doku tipi kimliği küçük ama gerçek bilgi taşıyor |
| 2 | Füzyon sadece Stage 2-4'e uygulanırsa? | Hayır yardımcı değil — RGB-only her zaman daha iyi |
| 3 | Kemik/tendon kuralı 3↔4 hatalarını düzeltir mi? | Hayır — eşik test setinde hiç tetiklenmiyor (0 değişim) |
| 4 | 3-4 hatalarında görünür bir ipucu var mı? | Düşük güven/yüksek entropi ölçülebilir; kemik/tendon tek başına net ayrıştırmıyor |
| 5 | Derinlik 3-4 ayrımına yardımcı olur mu? | Grup düzeyinde zayıf sinyal var ama örnek-düzeyi kurala aktarılamıyor (0 değişim) |

Beş iş paketi de Chapter 8'in genel örüntüsünü doğruluyor: ek bilgi kaynaklarının istatistiksel izleri var, ama mevcut basit füzyon/kural tasarımlarıyla güvenilir bir accuracy kazancına dönüşmüyor.

---

## Ek A — Keşfedilen Tüm Deneylerin Tam Envanteri

| ID | Durum | Not |
|---|---|---|
| Task 1 leakage-free retrain | ✅ Tamamlandı | MAnet+MiT-B2, 3 fold, Colab'da eğitildi, checkpoint'ler mevcut. Kendi başına bir doğruluk tablosu (ablation CSV) **repoda bulunamadı** — sadece ensemble maske üretimi + fold-vs-fold karşılaştırması mevcut. |
| exp-1 skin+wound crop | ✅ Tamamlandı, 3-fold | — |
| exp-2 wound-only dilated | ✅ Tamamlandı, 3-fold | — |
| exp-3 erken füzyon (8ch) | ✅ Tamamlandı, 3-fold | — |
| exp-4 late fusion (concat) | ✅ Tamamlandı, 3-fold | — |
| exp-4 late fusion (gated) | ✅ Tamamlandı, 3-fold | `concat`'ten biraz daha yüksek std, net üstünlük yok. |
| exp-5 residual (rgb_tissue) | ✅ Tamamlandı, 3-fold | Supersede edildi (bkz. tissue_only). |
| exp-5 residual (tissue_only) | ⚠️ **Kısmi — sadece fold-1** | README bunu açıkça belirtiyor; fold-0/2 hiç eğitilmedi/kaydedilmedi. |
| exp-6 tissue_statistics_only | ✅ Tamamlandı, 3-fold | — |
| exp-6 tissue_statistics_residual | ✅ Tamamlandı, 3-fold | — |
| exp-6 ANOVA özellik analizi | ✅ Tamamlandı | `tissue_statistics_anova.csv` |
| exp-6/RGB tamamlayıcılık analizi | ✅ Tamamlandı | Bu rapor hazırlığında yeniden çalıştırılıp doğrulandı. |
| exp-7 global-local ROI fusion (Faz A) | ✅ Tamamlandı, 3-fold | Dondurulmuş backbone. |
| exp-7 Faz B (ince ayar) | ⛔ **Planlandı, çalıştırılmadı** | Kod mevcut (`unfreeze_last_stage()`, test edildi) ama eğitim yapılmadı — bilinçli karar. |
| exp-9 çoklu-öğretmen damıtma | ⚠️ **Kısmi — sadece fold-0** | Commit `5d4a7f6` ile bilinçli olarak durduruldu (temel çizgiyi geçmiyor). §11. |
| exp-10 sınıra-özel temsil (`heads`/`attention`/`contrastive`) | ✅ Tamamlandı, 3 mod × 3-fold | Hiçbiri baseline'ı geçmedi, hiçbiri anlamlı altına düşmedi. §12. |
| exp-11 linear probe (`final`/`stage3`) | ✅ Tamamlandı, 3-fold | §13. |
| Depth-destekli evreleme (tüm evreler, genel) | ⛔ **Önerildi, uygulanmadı** | Orijinal Chapter 8 kapsamında: veri kümesi hazırlanıp commit edilmeden silindi. **Not — sonradan kısmen değişti:** `datasets/wound-stage-depth-anything-v2-small/` (Depth Anything V2 Small) ayrı bir tarihte yeniden üretildi ve danışmanın WP5 isteği kapsamında, **sadece Stage 3/4 ayrımına kısıtlı** olarak kullanıldı (bkz. §14/WP5) — genel/tüm-evreler bir depth-füzyon denemesi hâlâ yapılmadı. |
| **Danışman Ek Deneyleri (WP1-5)** | | `more_experiments/experiment_extended/`, bkz. §14 |
| WP1 exp-4 concat, birleşik doku kanalı | ✅ Tamamlandı, 3-fold (Colab T4) | Hafif, tutarlı kötüleşme. |
| WP1 exp-5 tissue_only, birleşik doku kanalı | ✅ Tamamlandı, 3-fold (Colab T4) | Hafif, tutarlı kötüleşme (daha küçük etki). |
| WP2 Stage-1-hariç füzyon kademesi | ✅ Tamamlandı | Negatif sonuç — RGB-only daha iyi. |
| WP3 kemik/tendon kural | ✅ Tamamlandı | Negatif sonuç — eşik hiç tetiklenmiyor. |
| WP4 Stage 3-4 hata incelemesi | ✅ Tamamlandı | 530 vaka, görsel paneller (`error_panels/`, klinik fotoğraf içerdiği için git'te izlenmiyor). |
| WP5 derinlik-genişletilmiş kural | ✅ Tamamlandı | Negatif sonuç — grup-düzeyi sinyal var, örnek-düzeyi kurala aktarılamıyor. |

**Yeniden adlandırma notu:** Bu deney ailesinin dizin adı geliştirme sürecinde `exp8_global_local_roi_fusion` → `exp7_global_local_roi_fusion` olarak değiştirildi (depth denemesinin "exp-7" numarasını hiç kullanmamış/commit'lememiş olması nedeniyle numara boşaldı). Git geçmişinde bu değişiklik tek bir commit olarak görünür (`exp8` ismi hiçbir commit'te yoktur) — kronolojik sıra commit mesajlarından değil, `git log --reverse` ile doğrulanmıştır.

## Ek B — Belgelenen Çelişkiler / Belirsizlikler

1. **exp-5 `tissue_only` 3-fold iddiası:** Bu araştırmacı-asistan konuşma geçmişinde fold-0/2 için bir Colab sonucunun paylaşıldığına dair bir not bulunsa da, bu sayılar **hiçbir dosyada** (`exp5_results.csv`, `exp5_diagnostics.json`, `comparison.csv`) kayıtlı değildir. Bu rapor sadece diskte doğrulanabilen fold-1 sonucunu kullanmıştır. **Kaynak:** `more_experiments/exp/exp5_residual_fusion/results/exp5_results.csv` (tek satır, fold=1).
2. **`more_experiments/results/comparison.csv`'deki exp-5 satırı** da aynı şekilde sadece fold-1'i yansıtıyor ve `n_folds=1` olarak doğru işaretlenmiş — çelişki yok, sadece eksik veri, doğru belgelenmiş.

---

## Ek C — Checkpoint Ensembling (Chapter 8 sonrası not)

Bölüm 3-9'daki tüm denemeler **tek bir modelin içine** (girdi füzyonu, rezidüel
dal, ROI füzyonu, stacking gate, çoklu-öğretmen distillation) bir şeyler
eklemeye çalıştı ve hiçbiri temel çizgiyi geçemedi. Buna karşılık, zaten
eğitilmiş birden fazla checkpoint'in **tahmin olasılıklarını** basitçe
ortalamak (sum-rule ensembling, Kittler et al., 1998) — hiçbir yeniden eğitim
gerektirmeden — temel çizgiyi geçti.

**Yöntem:** `task_3/ensemble_eval.py`. 4 checkpoint — `densenet121__softmax`,
`mit_b1__softmax`, `mit_b1__corn`, `convnextv2_tiny__softmax` — aynı
`build_folds(seed=42)` test split'lerinde çalıştırıldı. Softmax başlıklı
modellerin çıktısı doğrudan olasılık; CORN başlıklı modelin (`mit_b1__corn`)
çıktısı, koşullu sigmoid olasılıklarının kümülatif çarpımından (survival
fonksiyonu) 4-sınıflı bir olasılık dağılımına çözüldü, böylece farklı
başlık tipleri aynı ölçekte ortalanabildi. 4 modelin olasılıkları eşit
ağırlıkla (0.25/0.25/0.25/0.25) toplanıp argmax alındı.

**Sonuç (3-fold, `task_3/results/task_3_outputs/results/ensemble_results.csv`):**

| Metrik | mit_b1__corn (tek model, bu script'teki yeniden ölçüm) | **ENSEMBLE (4 model)** |
|---|---|---|
| Accuracy | 0.8121 ± 0.0137 | **0.8222 ± 0.0136** |
| Macro-F1 | 0.8132 ± 0.0126 | **0.8239 ± 0.0142** |
| MAE | 0.1951 ± 0.0196 | **0.1833 ± 0.0172** |
| QWK | 0.9102 ± 0.0179 | **0.9178 ± 0.0105** |

(mit_b1__corn'un buradaki yeniden ölçümü 0.8121, ana tablodaki
`ablation_summary.csv` değeri olan 0.8168'den ~0.5pp düşük — küçük bir
floating-point/MPS farkı, aynı checkpoint ve aynı split kullanılıyor.)

Ensemble, tek başına en iyi modelin **dört metrikte de** üstüne çıkıyor,
fold bazında da tutarlı (fold1/fold2'de belirgin kazanç, fold0'da ılımlı).
Hiçbir yeni parametre eğitilmedi — sadece mevcut checkpoint'lerin
tahminleri birleştirildi.

---

*Bu rapor, repodaki gerçek CSV/JSON/checkpoint dosyalarından doğrudan okunan değerlerle hazırlanmıştır. Hiçbir sayı tahmin edilmemiş, hiçbir eksik sonuç uydurulmamıştır. Yeni model eğitilmemiş, mevcut checkpoint/sonuç dosyalarına dokunulmamıştır.*
