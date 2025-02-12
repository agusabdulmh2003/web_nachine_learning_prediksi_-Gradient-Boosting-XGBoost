import streamlit as st
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---- KONFIGURASI UI ----
st.set_page_config(page_title="Prediksi Kangkung Sonic Bloom", layout="wide")

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .title {text-align: center; font-size: 36px; color: #1F4E79;}
    .subtitle {text-align: center; font-size: 22px; color: #3A506B;}
    .stButton>button {width: 100%; background-color: #1F4E79; color: white; font-size: 18px; padding: 10px; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("<h1 class='title'>🌱 Prediksi Pertumbuhan Kangkung Hidroponik Sonic Bloom</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analisis AI + Simulasi untuk Prediksi Masa Panen Kangkung</p>", unsafe_allow_html=True)

# ---- SIDEBAR (Input Data) ----
st.sidebar.header("📊 Masukkan Data Prediksi")

inputTinggi = st.sidebar.number_input("🌿 Tinggi Tanaman (cm):", min_value=0.0, format="%.2f")
inputSuhuair = st.sidebar.number_input("🌡️ Suhu Air (°C):", min_value=0.0, format="%.2f")
inputSuhuruang = st.sidebar.number_input("🏠 Suhu Ruang (°C):", min_value=0.0, format="%.2f")

jenis_musik_mapping = {'Dangdut': 0, 'Jazz': 1, 'Murottal': 2, 'Tanpa Musik': 3}
inputjenismusik = st.sidebar.selectbox("🎵 Jenis Musik:", list(jenis_musik_mapping.keys()))

predict_btn = st.sidebar.button("🔍 Prediksi Sekarang")

# ---- MEMUAT DATASET ----
dataset_path = 'dataset_sonicbloom.csv'
if os.path.exists(dataset_path):
    myData = pd.read_csv(dataset_path)

    # Menampilkan data dalam mode expandable
    with st.expander("📁 Lihat Dataset yang Digunakan"):
        st.dataframe(myData)

    # ---- PROSES MODEL ----
    X = myData[['tinggi', 'suhu_air', 'suhu_ruang', 'jenis_musik']]
    y = myData['hari_real']

    label_encoder = LabelEncoder()
    X['jenis_musik'] = label_encoder.fit_transform(X['jenis_musik'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi Model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    # ---- MENAMPILKAN HASIL PREDIKSI ----
    if predict_btn:
        input_data = np.array([[inputTinggi, inputSuhuair, inputSuhuruang, jenis_musik_mapping[inputjenismusik]]])
        input_df = pd.DataFrame(input_data, columns=X.columns)
        prediction = model.predict(input_df)

        st.markdown("### 🎯 **Hasil Prediksi**")
        st.success(f"🌱 **Umur Panen Kangkung: {int(prediction[0])} Hari**")

        # Menampilkan Hasil Evaluasi Model
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="⚡ Akurasi Training", value=f"{train_accuracy:.3f}")
        with col2:
            st.metric(label="🎯 Akurasi Testing", value=f"{test_accuracy:.3f}")

        # ---- SHAP (EXPLAINABILITY) ----
        st.write("### 🔍 Kenapa Hasilnya Seperti Ini?")
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig)

        # ---- SIMULASI PERTUMBUHAN ----
        st.write("### 📈 Simulasi Pertumbuhan Kangkung 🚀")
        days = np.arange(0, int(prediction[0]) + 1, 1)
        growth_curve = np.log1p(days) * inputTinggi * 0.8  

        fig_simulation = go.Figure()
        fig_simulation.add_trace(go.Scatter(x=days, y=growth_curve, mode='lines+markers', name='Prediksi Pertumbuhan'))
        fig_simulation.update_layout(title="📊 Proyeksi Pertumbuhan Kangkung",
                                     xaxis_title="Hari",
                                     yaxis_title="Tinggi (cm)",
                                     template="plotly_white")

        st.plotly_chart(fig_simulation)

else:
    st.error("⚠️ Harap unggah dataset terlebih dahulu.")
