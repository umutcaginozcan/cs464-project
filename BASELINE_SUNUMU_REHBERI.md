# Baseline Experiment (Exp-0) — Sunumu Rehberi

> **Slayt Süresi:** ~5-6 dakika | **Anahtarlar:** Paradox göster, motivasyon kur

---

## 🎯 Sunumun Hedefi
Baseline deneyinin **"şeyin neden normal machine learning işe yaramıyor"** göstermesi. Böylece izleyici Exp-1 ile Exp-4'teki iyileştirmelerin neden gerekli olduğunu anlayacak.

---

## Slayt 1: Problem Kurulması (1-1.5 dakika)

### Başlık
**"The Imbalance Problem: Why Standard ML Fails"**

### Görseller
- **`outputs/figures/01_class_distribution.png`** — Pie chart veya bar (5110 hastanın 249'u stroke)
  
### Konuş
```
"Bugün size bir veri seti problemi göstereceğim. 
5110 hastamız var, ama stroke olanlar sadece 249 kişi. 
Yani hastaların %95'i stroke yaşamadı, sadece %5'i yaşadı.

Şu soru: Eğer bir doktor size gelip 'Tüm hastalara 
"stroke yok" de' derse, ne kadar doğru olur?
→ %95 doğru! Ama... hiç stroke hastasını yakalayamaz.

İşte bugünün ana problemi bu."
```

---

## Slayt 2: Baseline Nedir? (1.5 dakika)

### Başlık
**"Exp-0: Standard ERM — No Imbalance Handling"**

### Formula
```
L = (1/N) Σ ℓ(yᵢ, f(xᵢ))

Açıklama: Her örnek eşit ağırlık
```

### Diagram (şu şekilde)
```
5110 örnek
│
├─ 4861 "No Stroke" (95%) ← Gradient'i domineerleniyor
│
└─ 249 "Stroke" (5%) ← Ihmal ediliyor
```

### Konuş
```
"Standart machine learning şöyle çalışır:
Her veri noktasını eşit şekilde öğrenme hedefi içine alırsınız.
Ancak veri setinizde %95 negatif örnek varsa,
model gradientı %95 negatif yönünde gider.
Sonuç? Model 'her zaman "stroke yok" de' öğrenir.
Çünkü bu strateji %95 doğruluk veriyor!"
```

### Kullanılan Modeller
```
6 model: 
- Logistic Regression
- Gaussian Naive Bayes  
- KNN (k=1,3,5,7,9)
- SVM (RBF)
- Random Forest
- XGBoost
```

---

## Slayt 3: Sonuçlar — Tablo (1.5 dakika)

### Başlık
**"Test Set Results: The Accuracy Paradox"**

### Tablo (CSV'den direkt al)

| Model | Accuracy | Recall | Precision | F1 | AUC-ROC | AUC-PR |
|-------|----------|--------|-----------|-----|---------|---------|
| **Logistic Reg** | **95.21%** | **0.02%** ⚠ | 100% | 0.039 | 0.84 | 0.26 |
| Gaussian NB | 31.12% | **98%** ⚠ | 6.52% | 0.12 | 0.79 | 0.15 |
| KNN (k=1) | 91.88% | 4% | 5.41% | 0.046 | 0.50 | 0.05 |
| Random Forest | 95.01% | 2% | 33.3% | 0.038 | 0.78 | 0.16 |
| XGBoost | 93.84% | 4% | 11.76% | 0.060 | 0.79 | 0.15 |

### Vurgulanacak Noktalar

```
❌ ACCURACY PARADOX:
   Logistic Regression %95 doğruluk (YÜKSEK) 
   Ama Recall = 0.02% (ÇÖPSE ÇÖP)
   
   → 5000 stroke hastasından 4999'unu kaçırıyor!
   → Klinisyen açısından işe yaramaz.

❌ GAUSSIAN NB'nin çırpması:
   Recall = 98% (harika) ama
   Precision = 6.52% (çoğu uyarı yalan)
   → "Herkese stroke risk" demek gibi
   
❌ F1 ortalaması: 0.04 civarı
   (0 = çöplük, 1 = mükemmel)
```

---

## Slayt 4: Confusion Matrix (1 dakika)

### Başlık
**"Why Recall Matters: The Clinical Cost"**

### Görsel
**`outputs/results/exp0_baseline/confusion_matrices.png`**

### Konuş
```
"Burada Logistic Regression'ın confusion matrix'ini görüyorsunuz.

True Positive (TP): Stroke buldu = 1 hasta
False Negative (FN): Stroke var ama kaçırdı = 49 hasta ⚠️

Klinik gerçeklik: 
- 1 hastayı yanlış alarm etmek → sigorta ödedi, test yaptırdı
- 49 hastayı kaçırmak → İnme yaşadı, ölüm riski ⚠️

False Negative'nin maliyeti sonsuz büyük. 
Bu yüzden Recall PARASAL METRIK değil, YAŞAM METRIKTIR."
```

---

## Slayt 5: ROC-PR Curves (1 dakika)

### Başlık
**"Model Performance: ROC vs PR Curves"**

### Görseller
- **`outputs/results/exp0_baseline/roc_curves.png`** (ROC)
- **`outputs/results/exp0_baseline/pr_curves.png`** (PR)

### Konuş (seç birine)

#### ROC Curve'den:
```
"ROC curve'ü görüyorsünüz. 
Random Forest ve XGBoost'un AUC-ROC ≈ 0.78
Bu sayı kötü değil görünüyor...

Ama dikkat edin: ROC curve'ü sınıf dengesizliğini 
maskeler. Çünkü True Negative'ler var (4800+ hasta).
TN çok olunca FPR düşük kalır, AUC sahte iyiye gösteriyor."
```

#### PR Curve'den (daha açıklayıcı):
```
"PR curve'ü başka hikaye anlatıyor.
AUC-PR = 0.15-0.26 (kötü).

PR curve neden daha doğru?
Dengesiz veri için Precision-Recall'a bakmak,
accuracy'ye bakmaktan daha güvenilir.
Çünkü positive class'a (stroke) fokus ediyor."
```

---

## Slayt 6: Sonuç & Motivasyon (1 dakika)

### Başlık
**"Why Baseline Fails: Motivation for Exp-1 through Exp-4"**

### 4 Sorun Vurgula
```
1️⃣ ACCURACY MISLEADING
   → %95 accuracy, %2 recall. Biri diğeri yalan.
   
2️⃣ STANDARD ERM UYGUNSUZ  
   → Class weight'i yok, resampling yok, threshold tuning yok
   
3️⃣ CLINICAL USELESS
   → Stroke hastalarını kaçırıyor
   
4️⃣ FN MALIYETI YÜKSEK
   → Bir FN = potansiyel ölüm
```

### Geçiş Cümlesi
```
"Bu baseline deneyimiz bize gösteriyor ki:
➜ Class imbalance sadece 'rahat' bir problem değil
➜ Standart ML algoritmaları otomatik çözmez
➜ Aktif imbalance handling gerekliyor

Sonraki 4 deney bunu test etmek için tasarlandı:
• Exp-1: Class Weighting (teorik fix)
• Exp-2: SMOTE (data augmentation)  
• Exp-3: ADASYN (adaptive sampling)
• Exp-4: Threshold Tuning (operating point seçimi)

Görürsünüz ki her biri baseline'ı iyileştiriyor."
```

---

## 📊 Teknik Hazırlık Checklist

Sunuma koymadan önce:
- [ ] `outputs/figures/01_class_distribution.png` download et
- [ ] `outputs/results/exp0_baseline/test_results.csv` aç → tablo yap (PowerPoint'e yapıştır)
- [ ] `outputs/results/exp0_baseline/confusion_matrices.png` slayta koy
- [ ] `outputs/results/exp0_baseline/roc_curves.png` slayta koy
- [ ] `outputs/results/exp0_baseline/pr_curves.png` slayta koy

---

## 💡 Sunumda Dikkat Etme Noktaları

### ✅ YAP
- **Paradoxu vurgula**: Accuracy yüksek ama Recall sıfır
- **Klinik gerçekliği konuş**: "Doktor açısından işe yaramaz"
- **Confusion matrix'teki FN'ye göster**: "Bu 49 kişi inme yaşadı"
- **Motivasyon kur**: "Sonraki deneylerde bunu çözeceğiz"

### ❌ YAPMA
- Deneysel detaylara girme (5-fold CV nasıl çalışır vb.)
- Tüm 6 modeli birer birer analiz etme → sıkıcı
- "Model A, Model B iyi, Model C kötü" − önemli değil
- Matematiksel formülleri fazla detaylı anlatma

### 🎯 Timing
- Slayt 1 (problem): 1-1.5 min
- Slayt 2 (baseline nedir): 1.5 min
- Slayt 3 (tablo): 1.5 min
- Slayt 4 (confusion): 1 min
- Slayt 5 (curves): 1 min
- Slayt 6 (sonuç): 1 min
- **Toplam: 6-7 dakika** ← izleyici sıkılmaz

---

## 🎬 Alternative: Kısa Versyon (3 dakika)

Eğer zamandan sıkıştırılırsan:

1. Class distribution + "neden problem?" (30 sn)
2. Tablo + Accuracy Paradox vurgula (60 sn)
3. Confusion matrix + FN maliyeti (60 sn)
4. "Exp-1-4 bunu çözüyor" (30 sn)

---

## 📌 Örnek Açılış

```
"Merhaba, ben Veli.

Uzun süredir malaria, kanser, stroke gibi hastalık 
tahmininde machine learning kullanıyoruz. 
Ama bir problem var:

Hastanın çoğu sağlıklı. 
Sadece %5'i hasta.

Şimdi size soru: 
Eğer bir model 'herkese sağlıklı' derse, 
ne kadar doğru olur?

%95.

Ama tüm hasta hastaları yakalayabilir mi?

Hayır. Hiç birini.

İşte bugünün konusu bu problemi anlamak ve çözmek."
```

Böyle başlarsanız izleyici immediately hale geliyor.
