"""
Example ingestion / cleaning scaffold for a CSV dataset.
Run this after downloading the Kaggle dataset into `data/raw/`.
"""
import os
import pandas as pd


RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


def find_csv():
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith('.csv'):
                return os.path.join(root, f)
    return None


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Example cleaning steps showing operations you'll demonstrate in portfolio
    # 1) Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # 2) Parse datetimes
    for col in df.columns:
        if 'date' in col or 'time' in col:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass

    # 3) Handle missing values (example)
    # - numeric columns: fill with median
    num_cols = df.select_dtypes(include=['number']).columns
    for c in num_cols:
        median = df[c].median()
        df[c] = df[c].fillna(median)

    # 4) Categorical columns: fill with 'unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df[c] = df[c].fillna('unknown')

    return df


def main():
    csv_path = find_csv()
    if not csv_path:
        print("No CSV found under data/raw/. Run scripts/fetch_kaggle.py first.")
        return

    print("Loading:", csv_path)
    df = pd.read_csv(csv_path)
    print("Initial rows, cols:", df.shape)

    df_clean = basic_clean(df)
    out_path = os.path.join(OUT_DIR, 'merged_clean.csv')
    df_clean.to_csv(out_path, index=False)
    print("Saved cleaned data to", out_path)


if __name__ == '__main__':
    main()
