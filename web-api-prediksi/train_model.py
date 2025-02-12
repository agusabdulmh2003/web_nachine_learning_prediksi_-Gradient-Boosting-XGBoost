import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1️⃣ Load Dataset
data = pd.read_csv("data_pertumbuhan_tanaman.csv")

# 2️⃣ Pilih Fitur dan Target
X = data[['tinggi_tanaman', 'suhu_air', 'suhu_ruang', 'ph_air', 'intensitas_cahaya']]
y = data['kategori_pertumbuhan']  # Label: 'baik', 'sedang', 'buruk'

# 3️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy:.2f}")

# 6️⃣ Simpan Model
joblib.dump(model, "model_ml.pkl")
print("Model disimpan sebagai model_ml.pkl")
