# Veri Analizi & Görselleştirme Projesi (Kaggle dataset)

Bu klasörde Kaggle'dan indirip temizleyebileceğin, analiz ve görselleştirme yapabileceğin bir scaffold bulunur.

Örnek hızlı başlangıç (NYC Airbnb dataset önerisi):

1. Kaggle API token oluştur ve `%USERPROFILE%\.kaggle\kaggle.json` içine koy.
2. Sanal ortam ve bağımlılıkları kur:
```powershell
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
```
3. Veri indir (örnek slug):
```powershell
python scripts/fetch_kaggle.py --slug "new-york-city/nyc-airbnb-open-data" --out data/raw
```
4. Temizleme iskeletini çalıştır:
```powershell
python scripts/ingest_example.py
```
5. Temizlenmiş veri `data/processed/merged_clean.csv` altında olacak — EDA ve görselleştirme için kullan.

Notlar:
- Ham verileri GitHub'a koyma (çoğu Kaggle verisi yeniden dağıtıma izin vermez). Bunun yerine `scripts/fetch_kaggle.py` kullan.
- `data/raw/` otomatik olarak `.gitignore`'a eklenmelidir.
