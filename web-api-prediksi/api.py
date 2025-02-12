from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1️⃣ Load Model
model = joblib.load("model_ml.pkl")

# 2️⃣ Inisialisasi FastAPI
app = FastAPI()

# 3️⃣ Skema Input
class PlantData(BaseModel):
    tinggi_tanaman: float
    suhu_air: float
    suhu_ruang: float
    ph_air: float
    intensitas_cahaya: float

# 4️⃣ Endpoint API
@app.post("/predict")
def predict(data: PlantData):
    try:
        # Konversi input ke array
        features = np.array([[data.tinggi_tanaman, data.suhu_air, data.suhu_ruang, data.ph_air, data.intensitas_cahaya]])
        
        # Prediksi dengan model
        prediction = model.predict(features)

        # Konversi hasil ke format Python
        return {"kategori_pertumbuhan": prediction[0].tolist()}
    
    except Exception as e:
        return {"error": str(e)}

# 5️⃣ Jalankan Server API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
