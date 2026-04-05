import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="NYC Airbnb Çözümlemesi", layout="wide")

# Path parsing relative to script
CURRENT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(CURRENT_DIR, '..', 'data', 'processed', 'merged_clean.csv')

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

st.title("🗽 New York City Airbnb Analiz & Görselleştirme Projesi")

df = load_data()

if df.empty:
    st.error(f"Veri bulunamadı: {DATA_PATH}. Lütfen önce scripts/fetch_kaggle.py ve ingest_example.py betiklerini çalıştırın.")
    st.stop()

st.markdown("""
Bu interaktif panelde, Kaggle'dan alınan New York City Airbnb 2019 verileri kullanılarak mahalle, oda tipi ve fiyata göre veri analizleri bulunmaktadır. Veriler Python ve Pandas, görselleştirmeler ise Plotly ile yapılmıştır.
""")

# Sidebar filters
st.sidebar.header("Filtreler")
neighbourhoods = st.sidebar.multiselect(
    "Semt (Borough) Seçin",
    options=df['neighbourhood_group'].dropna().unique(),
    default=df['neighbourhood_group'].dropna().unique()
)

price_min, price_max = int(df['price'].min()), int(df['price'].max())
if price_max > 1000:
    price_max = 1000 # Fiyatları biraz mantıklı sınıra çekelim 

price_range = st.sidebar.slider("Fiyat Aralığı ($)", min_value=0, max_value=2000, value=(0, 500))

filtered_df = df[
    (df['neighbourhood_group'].isin(neighbourhoods)) & 
    (df['price'] >= price_range[0]) & 
    (df['price'] <= price_range[1])
]

# Top KPI row
col1, col2, col3 = st.columns(3)
col1.metric("Toplam İlan", len(filtered_df))
col2.metric("Ortalama Fiyat", f"${filtered_df['price'].mean():.2f}")
col3.metric("Ortalama Yorum", f"{filtered_df['number_of_reviews'].mean():.1f}")

st.divider()

col_map, col_chart = st.columns([1, 1])

with col_map:
    st.subheader("Harita Üzerinde Airbnb Dağılımı")
    # Using scatter_mapbox or simple scatter_map (ploty express requires token for mapbox, map is open)
    # We take a sample to avoid freezing the browser if data is large
    sample_df = filtered_df.sample(min(2000, len(filtered_df)))
    fig_map = px.scatter_mapbox(
        sample_df, lat="latitude", lon="longitude", color="price", size="number_of_reviews",
        hover_name="name", hover_data=["neighbourhood", "room_type"],
        color_continuous_scale=px.colors.sequential.Plasma, zoom=9, height=500,
        mapbox_style="carto-positron"
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

with col_chart:
    st.subheader("Oda Tipine Göre Ortalama Fiyat")
    avg_price = filtered_df.groupby(["neighbourhood_group", "room_type"])["price"].mean().reset_index()
    fig_bar = px.bar(
        avg_price, x="neighbourhood_group", y="price", color="room_type",
        barmode="group", text_auto='.2s', title="Semte ve Oda Tipine Göre Fiyatlar", 
        labels={"price": "Ortalama Fiyat ($)", "neighbourhood_group": "Semt", "room_type": "Oda Tipi"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Fiyat vs Yorum Sayısı Dağılımı")
fig_scatter = px.scatter(
    filtered_df.sample(min(2000, len(filtered_df))), x="price", y="number_of_reviews", 
    color="neighbourhood_group", hover_data=["room_type"],
    labels={"price": "Fiyat ($)", "number_of_reviews": "Yorum Sayısı", "neighbourhood_group": "Semt"},
    title="Daha ucuz yerler daha fazla mı yorum alıyor?"
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.markdown("**Geliştirici Notu:** Bu proje Kaggle NYC Airbnb veri seti temel alınarak hızlıca veri temizleme, veri işleme ve Python ile interaktif görselleştirme yeteneklerini göstermek üzere hazırlanmıştır.")
