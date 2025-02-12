  Model XGBoost untuk Prediksi
Menggunakan XGBoost Regressor untuk memprediksi umur kangkung berdasarkan tinggi, suhu air, suhu ruang, dan jenis musik.
Dataset dibagi menjadi train (60%) dan test (40%).
Model dievaluasi dengan akurasi pada training & testing set.
  Interpretasi Model dengan SHAP
SHAP (SHapley Additive exPlanations) digunakan untuk menunjukkan fitur mana yang paling memengaruhi prediksi.
shap.summary_plot() menampilkan grafik yang menunjukkan dampak dari masing-masing fitur.
Contoh: Jika suhu air tinggi lebih banyak berkontribusi, maka grafik akan menunjukkan bahwa suhu sangat memengaruhi prediksi.
  Simulasi Pertumbuhan Kangkung
Menggunakan Plotly untuk membuat grafik pertumbuhan tanaman berdasarkan hari.
Fungsi logaritmik (np.log1p()) digunakan untuk mensimulasikan pertumbuhan alami tanaman.
Grafik menampilkan proyeksi tinggi tanaman setiap hari sampai hari yang diprediksi.
