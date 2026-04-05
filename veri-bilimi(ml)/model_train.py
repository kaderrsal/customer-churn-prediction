import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib

def load_data_and_features(filepath='cleaned_churn_data.csv'):
    """Veriyi yükler ve özellik (Feature/X) ile hedef değişkeni (Target/y) ayırır."""
    df = pd.read_csv(filepath)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y

def build_and_evaluate_model():
    """
    Birden fazla Makine Öğrenmesi modelini kurar, eğitir ve performanslarını karşılaştırır.
    Doğruluk oranını artırmak için veri ölçeklendirme (Scaling) ve gelişmiş algoritmalar kullanır.
    """
    print("info: Veri yükleniyor ve hazırlık yapılıyor...")
    X, y = load_data_and_features()
    
    # 1. Feature Engineering (Özellik Mühendisliği) - Kategorik Verileri Sayısallaştırma
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # 2. Veri Bölme (Train / Test Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # 3. Veri Ön İşleme (Veri Ölçeklendirme - Scaling)
    # Lojistik Regresyon gibi modeller için sayısal sütunların aynı ölçekte olması doğruluğu artırır.
    print("info: Sayısal veriler standartlaştırılıyor (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    # 3.5. Veri Dengesizliğini Giderme (SMOTE)
    # Ayrılan (Churn=1) müşteri verisi az olduğu için modeli bu konuda eğitmekte zorlanıyoruz.
    # SMOTE (Sentetik Aşırı Örnekleme) ile azınlık sınıfını sentetik verilerle çoğaltıp eşitliyoruz.
    print("info: SMOTE uygulanarak veri seti dengeleniyor (Azınlık sınıfı çoğaltılıyor)...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"info: SMOTE Öncesi Sınıf Dağılımı: \n{y_train.value_counts()}")
    print(f"info: SMOTE Sonrası Sınıf Dağılımı: \n{y_train_resampled.value_counts()}")
    
    # 4. Modellerin Tanımlanması
    # Daha yüksek doğruluk için farklı makine öğrenmesi algoritmaları deniyoruz:
    models = {
        "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest (Rastgele Orman)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting (GBM)": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_model_name = ""
    best_acc = 0
    best_y_pred = None
    
    print("\n" + "="*50)
    print("info: Modeller eğitiliyor ve karşılaştırılıyor...")
    print("="*50)
    
    # 5. Modellerin Eğitilmesi ve Karşılaştırılması
    for model_name, model in models.items():
        # Modeli SMOTE uygulanmış, dengeli veri ile eğit
        model.fit(X_train_resampled, y_train_resampled)
        
        # Tahminleme (Prediction) - Test verisi asla değiştirilmez (orijinal veriden test edilir)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] 
        
        # Metrikler
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        
        print(f"\n--- {model_name} Sonuçları ---")
        print(f"Accuracy (Doğruluk Oranı): {acc:.4f}")
        print(f"ROC-AUC Skoru:             {roc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = model_name
            best_y_pred = y_pred

    print("\n" + "="*50)
    print(f"EN İYİ MODEL: {best_model_name} (Doğruluk: {best_acc:.4f})")
    print("="*50)
    
    print(f"\n{best_model_name} için Sınıflandırma Analizi (Classification Report):")
    print(classification_report(y_test, best_y_pred))
    
    # 5.5. En İyi Modeli ve Scaler'ı Ara Yüz (Streamlit) İçin Kaydetme
    joblib.dump(models[best_model_name], 'best_churn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    # Orijinal veri şemasını bilmemiz gerektiği için columns kaydı
    joblib.dump(X_train_resampled.columns.tolist(), 'model_columns.pkl')
    print("\nsuccess: Ara yüz için en iyi model ve veri dönüştürücüler başarıyla kaydedildi!")
    
    # 6. En İyi Modelin Hata Matrisini Görselleştirme
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, best_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Hata Matrisi - En İyi Model ({best_model_name})')
    plt.xlabel('Modelin Tahmini (Predicted)')
    plt.ylabel('Gerçek Durum (Actual)')
    plt.tight_layout()
    import os
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    build_and_evaluate_model()
