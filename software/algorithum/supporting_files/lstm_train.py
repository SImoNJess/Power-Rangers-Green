# Re-import everything after code execution state reset
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Reshape
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

class PriceLSTMModel:
    def __init__(self, look_back=60, forecast_horizon=60, model_path="lstm_model.h5", scaler_path="scaler.pkl"):
        self.look_back = look_back
        self.forecast_horizon = forecast_horizon
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.scaler = MinMaxScaler()
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.look_back, 2)))
        model.add(Dense(self.forecast_horizon * 2))
        model.add(Reshape((self.forecast_horizon, 2)))
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_multistep_dataset(self, data):
        X, y = [], []
        for i in range(len(data) - self.look_back - self.forecast_horizon + 1):
            X.append(data[i:i + self.look_back])
            y.append(data[i + self.look_back:i + self.look_back + self.forecast_horizon])
        return np.array(X), np.array(y)

    def train(self, price_data, epochs=10, batch_size=16):
        scaled_data = self.scaler.fit_transform(price_data)
        X, y = self.create_multistep_dataset(scaled_data)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def predict_future(self, price_data):
        scaled_data = self.scaler.transform(price_data)
        last_window = scaled_data[-self.look_back:].reshape(1, self.look_back, 2)
        pred_scaled_seq = self.model.predict(last_window)
        return self.scaler.inverse_transform(pred_scaled_seq[0])

    def plot_forecast(self, price_data, pred_seq):
        real_buy = price_data[-self.forecast_horizon:, 0]
        real_sell = price_data[-self.forecast_horizon:, 1]
        future_buy = pred_seq[:, 0]
        future_sell = pred_seq[:, 1]
        time_axis = list(range(-self.forecast_horizon, 0)) + list(range(1, self.forecast_horizon + 1))
        buy_combined = np.concatenate([real_buy, future_buy])
        sell_combined = np.concatenate([real_sell, future_sell])

        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, buy_combined, marker='o', label='Buy Price (Real + Predicted)')
        plt.plot(time_axis, sell_combined, marker='o', label='Sell Price (Real + Predicted)')
        plt.axvline(x=0, color='gray', linestyle='--', label='Prediction Start')
        plt.title("Electricity Prices (Past 60 ticks + Forecast 60 ticks)")
        plt.xlabel("Tick Offset (0 = Now)")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("price_forecast_60.png")
        plt.show()

# Load data and train
df = pd.read_csv("all_history.txt", delim_whitespace=True)
price_data = df[["buy_price", "sell_price"]].astype(float).values

model = PriceLSTMModel()
model.train(price_data)

pred_seq = model.predict_future(price_data)
model.plot_forecast(price_data, pred_seq)
