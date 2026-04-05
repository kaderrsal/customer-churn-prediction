# 📉 Müşteri Kaybı (Customer Churn) Tahmini Modeli

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

Bu proje, telekomünikasyon sektöründeki **müşteri kaybı (churn)** problemini çözmek ve potansiyel olarak şirketi terk edecek riskli müşterileri henüz ayrılmadan önce tespit etmek amacıyla geliştirilmiş uçtan uca bir veri bilimi çalışmasıdır.

---

## 🎯 Proje Özeti
Şirketlerin en büyük maliyeti yeni müşteri kazanmaktır. Mevcut müşteriyi elde tutmak çok daha ulaşılabilir ve kârlı bir hedeftir. Bu doğrultuda projenin temel yapısı; müşterilerin geçmiş verilerini (fatura tutarları, üyelik süresi, kullandığı ek servisler) analiz ederek, **ayrılma eğiliminde olan (Churn=1)** müşterileri denetimli makine öğrenmesi algoritmalarıyla yakalamaktır.

## ⚖️ Teknik İyileştirme: Sınıf Dengesizliği (Data Imbalance) ve SMOTE Algoritması

Makine öğrenmesi modelleri her zaman dengeli veri setleri ile optimum şekilde çalışır. Verimizde müşterilerin sadece **%26**'sının şirketi terk ettiği görülmüştür. Bu durumda algoritma kolaya kaçarak ağırlıklı olarak "Bu müşteri ayrılmayacak (0)" demeye (ve yüksek bir genel doğruluk tutturmaya) meyilliydi. Sonuç olarak, asıl tespit etmemiz gereken *"ayrılacak riskli müşteriyi yakalama"* başarı oranımız (Recall oranı) ilk aşamada **%57**'de kalıyordu.

**🚀 Çözüm (SMOTE Uygulaması):**
Azınlık durumdaki "Ayrılan (1)" müşteri verilerini ele alarak Sentetik Azınlık Aşırı Örnekleme (SMOTE) uyguladık ve veriyi eğitim modülünde "4130'a 4130" olacak şekilde dengeledik. Ortaya çıkan dengeli veride modelimizin gerçek tehlikeleri yakalama (Recall) potansiyelini ciddi şekilde artırdık.

### 📈 Öncesi & Sonrası Performans Değişimi

| Metrik | Uygulanan Model / Yöntem | Riskli/Terk Eden Müşteriyi Tespit Etme (Recall - 1) | Genel Doğruluk Oranı (Accuracy) |
| :--- | :--- | :---: | :---: |
| **Orijinal (Dengesiz) Veri** | Lojistik Regresyon | **%57.0** ⚠️ | ~ %80.5 |
| **SMOTE Uygulanmış (Dengeli) Veri** | Random Forest_ | **%64.0** ⭐ | ~ %76.4 |

> *(Önemli Not: Genel metrik Accuracy (Doğruluk) SMOTE sonrasında gerilemiş gibi görünse de işletme bakımından risksiz bir müşteriyi riskli olarak işaretlemek düşük bir pürüzdür. Ancak asıl tehlike olan **"şirketten ayrılma eylemindeki müşteriyi gözden kaçırmak" (Recall düşüklüğü)**, büyük para kaybıdır. Yani SMOTE operasyonu ile modelin risk tespiti ve stratejik kullanımı güçlendirilmiştir.)*

---

## 📊 Proje Adımları ve Görsel Analizler

### 1️⃣ Veri Ön İşleme ve Temizleme (Data Preprocessing)
Gereksiz değişkenler (`customerID`) veri grubundan atıldı, tip dönüşümleri düzeltildi ve boş veriler atıldı.

<p align="center">
  <img src="assets/cleaned_sample.png" width="90%" alt="Temiz Sütunlar" />
</p>

### 2️⃣ Keşifsel Veri Analizi (EDA)
Kategorik ve sayılara odaklanılmış değişkenlerin dağılımları grafiklere döküldü, hedef (target) değişkene olan bağımlılıklar incelendi.

<p align="center">
  <img src="assets/churn_distribution.png" width="45%" alt="Terk Edenler" />
  <img src="assets/num_features_hist.png" width="45%" alt="Kullanimlar" />
</p>

### 3️⃣ Model Eğitimi (Training & Evaluation)
Ayrılan %20'lik test verisi ölçeklendi (`StandardScaler`). SMOTE yapılandırmasının akabinde Lojistik Regresyon, Random Forest ve Gradient Boosting modelleri test edildi, güncel ortamdaki en esnek sonuçları **Random Forest** üretmiştir.

<p align="center">
  <img src="assets/confusion_matrix.png" width="50%" alt="Hata Matrisi" />
</p>

---

## 🛠️ Kurulum ve Çalıştırma

Projeyi lokal bilgisayar ortamında denerken aşağıdaki komutları izleyebilirsiniz:

```bash
# 1. Gerekli açık kaynak modülleri (kütüphaneleri) yükleyin
pip install pandas scikit-learn matplotlib seaborn imbalanced-learn

# 2. Veri okuma ve temizleme operasyonu
python data_preparation.py

# 3. İstatistik veriler ve model rapor/grafikleri (Görsel EDA)
python data_overview.py

# 4. Modeli Eğitin, SMOTE dengeleyicisini devreye sokun (.pkl dosyalarınıza yazın)
python model_train.py
```
