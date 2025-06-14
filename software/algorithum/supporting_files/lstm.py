import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from scipy.stats import norm
import matplotlib.pyplot as plt

# === 1. 读取数据（自动识别表头） ===
df = pd.read_csv("all_history.txt", delim_whitespace=True)
price_data = df[["buy_price", "sell_price"]].astype(float).values

# === 2. 归一化处理 ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(price_data)

# === 3. 构造多步预测数据集 ===
def create_multistep_dataset(data, look_back=24, forecast_horizon=6):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back:i+look_back+forecast_horizon])
    return np.array(X), np.array(y)

look_back = 24
forecast_horizon = 6
X, y = create_multistep_dataset(scaled_data, look_back, forecast_horizon)

# === 4. 构建修复后的 LSTM 模型 ===
model = Sequential()
model.add(LSTM(64, input_shape=(look_back, 2)))                  # 输出为 (batch_size, 64)
model.add(Dense(forecast_horizon * 2))                           # 输出为 (batch_size, 12)
model.add(Reshape((forecast_horizon, 2)))                        # 输出为 (batch_size, 6, 2)
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=30, batch_size=16, verbose=1)

# === 5. 进行未来价格预测 ===
last_window = scaled_data[-look_back:].reshape(1, look_back, 2)
pred_scaled_seq = model.predict(last_window)
pred_seq = scaler.inverse_transform(pred_scaled_seq[0])  # shape: (6, 2)

future_buy = pred_seq[:, 0]
future_sell = pred_seq[:, 1]

# === 6. 获取真实价格用于对比画图 ===
real_buy = price_data[-6:, 0]
real_sell = price_data[-6:, 1]

time_axis = list(range(-6, 0)) + list(range(1, 7))
buy_combined = np.concatenate([real_buy, future_buy])
sell_combined = np.concatenate([real_sell, future_sell])

P_buy_mean = np.mean(future_buy)
P_sell_mean = np.mean(future_sell)

print("未来平均 P_buy：", round(P_buy_mean, 4))
print("未来平均 P_sell：", round(P_sell_mean, 4))

plt.figure(figsize=(10, 5))
plt.plot(time_axis, buy_combined, marker='o', label='Buy Price (Real + Predicted)')
plt.plot(time_axis, sell_combined, marker='o', label='Sell Price (Real + Predicted)')
plt.axvline(x=0, color='gray', linestyle='--', label='Prediction Start')
plt.title("Electricity Prices (Past 6h + Forecast 6h)")
plt.xlabel("Hour Offset (0 = Now)")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("price_forecast.png")
plt.show()

# === 7. 期望价值函数 ===
def expected_value(E, mu_D=1, sigma_D=1):
    prob_exceed = 1 - norm.cdf(E, loc=mu_D, scale=sigma_D)
    expected_gain = P_sell_mean * min(E, mu_D)
    expected_loss = P_buy_mean * prob_exceed * max(0, mu_D - E)
    return expected_gain - expected_loss

# === 8. 贪心策略决策函数 ===
def greedy_decision(E, current_P_buy, current_P_sell, step=1):
    value_gain_buy = expected_value(E + step) - expected_value(E)
    cost_buy = step * current_P_buy

    value_loss_sell = expected_value(E) - expected_value(E - step)
    revenue_sell = step * current_P_sell

    if value_gain_buy > cost_buy and E + step <= 10:
        return "buy"
    elif value_loss_sell < revenue_sell and E - step >= 0:
        return "sell"
    else:
        return "hold"

# === 9. 应用决策 ===
current_energy_level = 5

current_P_buy = price_data[-1, 0]   # 最新一小时的买入价
current_P_sell = price_data[-1, 1]  # 最新一小时的卖出价
decision = greedy_decision(current_energy_level, current_P_buy, current_P_sell)
print("当前电量 =", current_energy_level, "→ 贪心策略决策 3=", decision)