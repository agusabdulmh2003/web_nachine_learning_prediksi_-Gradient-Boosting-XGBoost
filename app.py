import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # Menggunakan XGBoost
from sklearn.preprocessing import LabelEncoder

# Menampilkan judul aplikasi
st.title("Program Prediksi Pertumbuhan Kangkung Hidroponik Sonic Bloom")

# Menampilkan gambar jika tersedia
image_path = './sonicbloom.png'
if os.path.exists(image_path):
    st.image(image_path)
else:
    st.warning("Gambar tidak ditemukan.")

# Deskripsi Sonic Bloom
st.write("""
## Sonic Bloom
Sonic Bloom adalah pancaran frekuensi tinggi seperti musik yang digunakan untuk memperlebar stomata pada daun.
""")

st.write("## Keterangan Data yang Digunakan")
st.markdown("""
1. **Tinggi**: Tinggi tumbuhan kangkung hidroponik saat prediksi dilakukan.
2. **Suhu Air**: Suhu air saat prediksi.
3. **Suhu Ruang**: Suhu ruang saat prediksi.
4. **Jenis Musik**: Musik yang digunakan dalam perlakuan Sonic Bloom.
""")

# Membaca dataset
dataset_path = 'dataset_sonicbloom.csv'
if os.path.exists(dataset_path):
    myData = pd.read_csv(dataset_path)
    st.write("## Overview Data")
    st.dataframe(myData)

    st.write("## Deskripsi Data")
    st.dataframe(myData.describe())

    # Memisahkan fitur dan label
    X = myData[['tinggi', 'suhu_air', 'suhu_ruang', 'jenis_musik']]
    y = myData['hari_real']

    # Mengubah jenis musik menjadi numerik
    label_encoder = LabelEncoder()
    X = X.copy()
    X['jenis_musik'] = label_encoder.fit_transform(X['jenis_musik'])

    st.write("## Input Data X", X.head())
    st.write("## Label Data y", y.head())

    # Split data untuk training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Model XGBoost
    model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model_xgb.fit(X_train, y_train)

    # Evaluasi model
    train_accuracy = model_xgb.score(X_train, y_train)
    test_accuracy = model_xgb.score(X_test, y_test)
    
    st.write("### Evaluasi Model XGBoost")
    st.write(f"On Train Accuracy: {train_accuracy:.3f}")
    st.write(f"On Test Accuracy: {test_accuracy:.3f}")

    # Form untuk prediksi
    st.write("## Masukkan Data untuk Prediksi")
    
    form = st.form(key='prediction_form')
    inputTinggi = form.number_input("Masukkan tinggi tumbuhan (cm):", min_value=0.0, format="%.2f")
    inputSuhuair = form.number_input("Masukkan suhu air (°C):", min_value=0.0, format="%.2f")
    inputSuhuruang = form.number_input("Masukkan suhu ruang (°C):", min_value=0.0, format="%.2f")
    
    jenis_musik_mapping = {'Dangdut': 0, 'Jazz': 1, 'Murottal': 2, 'Tanpa Musik': 3}
    inputjenismusik = form.selectbox("Pilih jenis musik:", list(jenis_musik_mapping.keys()))
    
    submit = form.form_submit_button('Prediksi')

    if submit:
        # Konversi input ke bentuk numerik
        input_data = np.array([[inputTinggi, inputSuhuair, inputSuhuruang, jenis_musik_mapping[inputjenismusik]]])
        input_df = pd.DataFrame(input_data, columns=X.columns)

        # Prediksi umur kangkung
        prediction = model_xgb.predict(input_df)
        st.success(f"Prediksi umur tumbuhan kangkung hidroponik: **{int(prediction[0])} hari**")
else:
    st.error("Dataset tidak ditemukan. Harap unggah file dataset_sonicbloom.csv.")
