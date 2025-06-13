import numpy as np
from tensorflow.keras.models import load_model
import joblib

# load
MODEL  = load_model("lstm_model.h5", compile=False)
SCALER = joblib.load("scaler.pkl")

# dummy “yesterday data”
arr = np.random.rand(60,2)  # or load your real yesterday_data
scaled = SCALER.transform(arr)
pred_scaled = MODEL.predict(scaled.reshape(1,60,2))[0]
pred = SCALER.inverse_transform(pred_scaled)

# inspect
print("IN  :", arr[:5])
print("OUT :", pred[:5])
