import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(filepath='cleaned_churn_data.csv'):
    """
    Temizlenmiş verinin Keşifsel Veri Analizi'ni (EDA - Exploratory Data Analysis) yapar.
    Verinin yapısını, istatistiklerini ve dağılımını anlamamızı sağlar.
    """
    print("info: Keşifsel Veri Analizi (EDA) başlatılıyor...")
    df = pd.read_csv(filepath)
    
    # 1. Veri Seti Hakkında Genel Bilgiler
    print("\n--- Veri Tipleri ---")
    print(df.dtypes)
    
    print("\n--- Temel İstatistiksel Özet (Sayısal Veriler) ---")
    print(df.describe())
    
    print("\n--- Hedef Değişken (Churn) Dağılım Oranı ---")
    # Kaç müşteri şirketi terk etmiş (1) / terk etmemiş (0) oransal gösterim.
    print(df['Churn'].value_counts(normalize=True) * 100)
    
    # 2. Kategorik (Metinsel) Özelliklerin İncelenmesi
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        print(f"\n--- '{col}' Sütunu Frekans Dağılımı ---")
        print(df[col].value_counts())
    
    # 3. Analiz Görselleştirme (Data Visualization)
    # 3.a. Churn oranının çubuk grafiği (Bar chart)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, palette='viridis')
    plt.title('Müşteri Kaybı (Churn) Dağılımı (0: Kayıp Yok, 1: Kayıp Var)')
    plt.xlabel('Churn Durumu')
    plt.ylabel('Müşteri Sayısı')
    plt.show()
    
    # 3.b. Sayısal değişkenlerin (TotalCharges, MonthlyCharges vb.) histogramları
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols].hist(figsize=(12, 8), bins=30, color='skyblue', edgecolor='black')
    plt.suptitle('Sayısal Değişkenlerin Dağılımları', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_data()

