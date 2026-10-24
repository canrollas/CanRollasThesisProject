# Chapter 8 — 5 Dakikalık Sunum Taslağı

Amaç: danışmana implementasyon detaylarına boğulmadan, en önemli bulguları ve "neden" sorusunu anlatmak. Ayrıntı için `supervisor_experiment_summary.md`'ye referans ver.

---

### 0:00–0:45 — Soru ve temel çizgi (1 slayt)

- "RGB MiT-B1+CORN zaten güçlü: 0.8168 accuracy, 0.9138 QWK, 3-fold CV."
- "Soru: elimizdeki Task 1 (lokalizasyon) ve Task 2 (doku segmentasyonu) modelleri bunu geliştirebilir mi?"
- "Yedi deney ailesi, tutarlı aynı PIID 3-fold protokolüyle test edildi."

### 0:45–1:45 — Lokalizasyon denemeleri: negatif ama öğretici (1 slayt, `accuracy_comparison.png`'nin ilk 3 çubuğu)

- exp-1 (deri+yara maskesi) ve exp-2 (sadece yara, adaptif): ikisi de ~3.3pp düşüş, **birbirinden ayırt edilemez**.
- Mesaj: sorun "ne kadar maskelendiği" değil, maskelemenin kendisi — arka plan/bağlam gerçek sinyal taşıyordu.

### 1:45–3:00 — Doku füzyonu: nasıl birleştirdiğin önemli (2 slayt)

- Erken füzyon (exp-3, 8-kanal): temel çizginin altında.
- Geç füzyon (exp-4): belirgin iyileşme, QWK'de temel çizgi std'si içine giriyor — **ilk gerçek sinyal**.
- Rezidüel füzyon (exp-5/6), dondurulmuş gerçek temel çizgiyle: temel çizgiyle istatistiksel olarak aynı.
- **Teşhis bulgusu (en önemli slayt):** shuffle-ablasyonu, rezidüel kolun gerçek doku içeriğini kullanmadığını gösterdi (kosinüs≈1.0) — "temel çizgiyi eşlemek" ile "gerçekten doku öğrenmek" aynı şey değilmiş.

### 3:00–3:45 — Tamamlayıcılık analizi: sinyal var, çıkarılamıyor (1 slayt, `complementarity_contingency.png`)

- Doku istatistikleri tek başına şans seviyesinin üzerinde (%38 vs %25-29).
- RGB yanlış olduğunda tissue %43.5 doğru — gerçek koşullu sinyal.
- Oracle üst sınırı %89.6 — ama bu ulaşılamaz bir teorik tavan, gerçek model değil.
- **Mesaj:** doku bilgisi var, mevcut füzyon mimarileri onu yakalayamıyor.

### 3:45–4:30 — Global-local ROI füzyonu: en net negatif sonuç (1 slayt, `architecture_diagram.png` son panel)

- Literatürden (Bayraktar ve ark. 2026) uyarlanmış, sıfır-başlatmalı, temel-çizgi-koruyucu mimari.
- ROI ablasyonu: tahmini/tam-görüntü/karıştırılmış kutu arasında **0 tahmin farkı**, 3 fold'un hepsinde.
- **Önemli nüans:** bu "lokalizasyon işe yaramaz" demek değil — bu *spesifik* dondurulmuş-backbone tasarımının sinyali çıkaramadığı demek. Faz B (ince ayar) hiç denenmedi.

### 4:30–5:00 — Kapanış (1 slayt)

- Hiçbir füzyon RGB temel çizgisini istatistiksel olarak geçmedi.
- Ama: doku bilgisinin gerçek olduğu doğrudan kanıtlandı (tamamlayıcılık analizi) — sorun bilginin yokluğu değil, çıkarım mimarisinin yetersizliği.
- Somut gelecek yön: RGB kolunun dondurulmadığı uçtan uca ince ayar, ya da bilginin backbone içine erken katmanlarda entegre edilmesi.

---

**Olası sorulara hazırlık:**
- *"Neden hiçbiri işe yaramadı?"* → Teşhis scriptleri (gradyan normu, shuffle-ablasyonu) kök nedeni gösterdi: ek dallar yeterince güçlü bir öğrenme sinyali almıyor, RGB yolu zaten çok güçlü.
- *"Tek-fold sonuçlara güvenilir mi?"* → Hayır, raporda açıkça kısmi olarak işaretlendi (exp-5 tissue-only), 3-fold sonuçlarla karıştırılmadı.
- *"Bir sonraki adım ne olurdu?"* → Faz B ince ayar (exp-7) veya RGB kolunun uçtan uca eğitildiği bir tasarım — ikisi de bu chapter'da bilinçli olarak denenmedi.
