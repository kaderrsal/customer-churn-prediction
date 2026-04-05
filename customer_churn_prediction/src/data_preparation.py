import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_clean_data(filepath='data/churn_data.csv'):
    """
    Ham müşteri kaybı (churn) verisini yükler, gereksiz kolonları temizler, 
    veri tiplerini düzeltir ve işlenmiş veriyi dışa aktarır.
    
    Argümanlar:
        filepath (str): Ham verinin bulunduğu dosya yolu.
        
    Döndürür:
        pd.DataFrame: Temizlenmiş veri seti.
    """
    print("info: Veri seti yükleniyor...")
    df = pd.read_csv(filepath)
    print(f"info: Orijinal veri boyutu: {df.shape}")
    
    # 1. Gereksiz Verilerin Çıkarılması:
    # 'customerID' modeli eğitirken hiçbir bilgi taşımaz (rastgele bir kimliktir).
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        print("info: 'customerID' sütunu veri setinden kaldırıldı.")
    
    # 2. Veri Tiplerinin Düzeltilmesi:
    # 'TotalCharges' (Toplam Ücret) bilgisayar tarafından yanlışlıkla metin (string) olarak
    # algılanmış olabilir. Bunu zorla sayısal tipe (numeric) çeviriyoruz.
    # errors='coerce' parametresi, sayıya çevrilemeyen boşluk veya hatalı metinleri NaN (Not a Number) yapar.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 3. Eksik Veri (Missing Value) Yönetimi:
    missing_tot = df['TotalCharges'].isnull().sum()
    print(f"warning: 'TotalCharges' kolonunda {missing_tot} adet eksik (boş) veri bulundu.")
    
    # Eksik veri sayısı çok az (örn. 11 satır) olduğu için bu satırları tamamen siliyoruz (Drop).
    df.dropna(inplace=True)
    print(f"info: Eksik veriler kaldırıldıktan sonra veri boyutu: {df.shape}")
    
    # 4. Hedef Değişkenin (Target) Dönüştürülmesi:
    # Makine öğrenmesi modelleri sayısal değerlerle çalışır. 'Yes'->1, 'No'->0 olarak kodluyoruz.
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # İşlenmiş veriyi yeni bir CSV dosyası olarak kaydetme işlemi.
    cleaned_filepath = 'data/cleaned_churn_data.csv'
    df.to_csv(cleaned_filepath, index=False)
    print(f"success: Temizlenmiş veri '{cleaned_filepath}' olarak kaydedildi.")
    # Temizlenmiş verinin ilk 5 satırını görselleştirip docs/images klasörüne kaydet
    os.makedirs('docs/images', exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.axis('off')
    tbl = ax.table(cellText=df.head().values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig('docs/images/cleaned_sample.png')
    plt.close()
    return df

if __name__ == "__main__":
    # Bu dosya doğrudan çalıştırıldığında (import edilmediğinde) aşağıdaki kod tetiklenir.
    df_cleaned = load_and_clean_data()

