# Re-import necessary libraries after code execution state reset
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import numpy as np
print("test 2")
# === Load model and scaler ===
model = load_model("lstm_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# === Load historical price data ===
df = pd.read_csv("all_history.txt", delim_whitespace=True)
price_data = df[["buy_price", "sell_price"]].astype(float).values

look_back = 60
forecast_horizon = 60

# === Randomly select a valid index for prediction ===
start_index = np.random.randint(0, len(price_data) - look_back - forecast_horizon)
input_seq = price_data[start_index:start_index + look_back]
true_future = price_data[start_index + look_back: start_index + look_back + forecast_horizon]

# === Predict using LSTM ===
scaled_input = scaler.transform(input_seq)
lstm_input = scaled_input.reshape(1, look_back, 2)
pred_scaled = model.predict(lstm_input)
predicted_future = scaler.inverse_transform(pred_scaled[0])

# === Plotting buy price comparison ===
true_buy = true_future[:, 0]
pred_buy = predicted_future[:, 0]

plt.figure(figsize=(10, 5))
plt.plot(range(forecast_horizon), true_buy, label="Actual Buy", linewidth=3, color="green")
plt.plot(range(forecast_horizon), pred_buy, label="Predicted Buy", linestyle="--", linewidth=2, color="lightgreen")
plt.title("Buy Price Comparison (Random 60-Tick Window)")
plt.xlabel("Tick")
plt.ylabel("Buy Price (p)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("buy_price_comparison_random_test.png")
plt.show()
