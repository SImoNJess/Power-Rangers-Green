import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import MinMaxScaler
from transformer_policy import TransformerPolicy

print("Evaluating PPO TransformerPolicy vs Price-Based Strategies (30-Day Episodes)")

# === Constants ===
STATE_SEQ_LEN = 60
STATE_DIM = 4
ACTION_SIZE = 17
MAX_STORAGE = 50
MAX_POWER = 4.0
TICK_DURATION = 5
EPISODES = 100
TOTAL_TICKS_PER_EPISODE = 60 * 30
action_space = np.linspace(-40.0, 40.0, ACTION_SIZE)

# === Load model and scaler ===
model = tf.keras.models.load_model("ppo_trader_policy.keras", custom_objects={"TransformerPolicy": TransformerPolicy})
scaler = joblib.load("scaler_dqn.pkl")

# === Load and preprocess data ===
df = pd.read_csv("all_history_full.txt", sep="\s+")
tick_column = df.iloc[:, 1].values
raw_data = df[["sun", "price_buy", "price_sell", "demand"]].astype(float).values
pv_raw = np.minimum(raw_data[:, 0], 100.0) * 0.032
features = scaler.transform(uniform_filter1d(raw_data, size=5, axis=0))
valid_starts = np.where(tick_column == 0)[0]

# === Policy Functions ===
def constant_zero_policy(state_seq, t):
    return 8  # action index for 0.0

def price_threshold_policy(state_seq, t, threshold=1.0):
    sell_price = state_seq[t][1]
    buy_price = state_seq[t][2]
    sell_threshold = 40
    buy_threshold = 30
    if sell_price > sell_threshold:
        return 0
    elif buy_price < buy_threshold:
        return 16
    else:
        return 8

# === Episode Simulator ===
def run_episode(full_raw_seq, full_scaled_seq, full_pv_seq, model=None, seed_storage=0, custom_policy_fn=None, debug=False):
    storage = seed_storage
    total_cost = 0
    action_log = []
    full_raw_seq=full_scaled_seq
    for t in range(TOTAL_TICKS_PER_EPISODE):
        cur_window = full_scaled_seq[t:t + STATE_SEQ_LEN]
        if cur_window.shape[0] < STATE_SEQ_LEN:
            pad = np.tile(cur_window[-1], (STATE_SEQ_LEN - cur_window.shape[0], 1))
            cur_window = np.vstack([pad, cur_window])
        x_seq = np.expand_dims(cur_window, axis=0).astype(np.float32)
        x_storage = np.array([[storage / MAX_STORAGE]], dtype=np.float32)

        if custom_policy_fn:
            action_index = custom_policy_fn(full_raw_seq, t)
            if debug and t < 60:
                print(f"[Tick {t}] Price_buy={full_raw_seq[t][1]}, Price_sell={full_raw_seq[t][2]} → Action Index = {action_index}")
            action_log.append(action_index)
        else:
            logits, _ = model([x_seq, x_storage], training=False)
            action_probs = tf.nn.softmax(logits)
            dist = tf.squeeze(action_probs)
            action_index = tf.random.categorical(tf.math.log([dist]), 1)[0, 0].numpy()

        action = action_space[action_index]

        sun, price_sell, price_buy, demand_scaled = full_raw_seq[t]
        pv = full_pv_seq[t] * TICK_DURATION
        demand = demand_scaled * MAX_POWER * TICK_DURATION

        serve = demand
        use_pv = min(pv, serve)
        serve -= use_pv
        remaining_pv = pv - use_pv

        use_battery = min(storage, serve)
        storage -= use_battery
        serve -= use_battery

        cost = 0
        if serve > 0:
            cost += serve * price_buy

        if remaining_pv > 0 and storage < MAX_STORAGE:
            store_pv = min(MAX_STORAGE - storage, remaining_pv)
            storage += store_pv
            remaining_pv -= store_pv

        if remaining_pv > 0:
            cost -= remaining_pv * price_sell

        allowed = np.clip(action, -storage, MAX_STORAGE - storage)
        storage += allowed
        if allowed > 0:
            cost += allowed * price_buy
        else:
            cost -= -allowed * price_sell

        total_cost += cost

    return total_cost, action_log

# === Evaluation ===
ppo_costs, zero_costs, threshold_costs = [], [], []

# Sample a single episode to inspect threshold behavior
idx = np.random.choice(valid_starts[valid_starts <= len(features) - TOTAL_TICKS_PER_EPISODE])
full_raw_seq = raw_data[idx:idx + TOTAL_TICKS_PER_EPISODE]
full_scaled_seq = features[idx:idx + TOTAL_TICKS_PER_EPISODE]
full_pv_seq = pv_raw[idx:idx + TOTAL_TICKS_PER_EPISODE]

print("🧪 Sample threshold policy behavior:")
_, threshold_action_log = run_episode(full_raw_seq, full_scaled_seq, full_pv_seq,
                                      custom_policy_fn=lambda s, t: price_threshold_policy(s, t, threshold=1.0),
                                      debug=True)

# Run full evaluation
for i in range(EPISODES):
    idx = np.random.choice(valid_starts[valid_starts <= len(features) - TOTAL_TICKS_PER_EPISODE])
    full_raw_seq = raw_data[idx:idx + TOTAL_TICKS_PER_EPISODE]
    full_scaled_seq = features[idx:idx + TOTAL_TICKS_PER_EPISODE]
    full_pv_seq = pv_raw[idx:idx + TOTAL_TICKS_PER_EPISODE]

    zero_costs.append(run_episode(full_raw_seq, full_scaled_seq, full_pv_seq, custom_policy_fn=constant_zero_policy)[0])
    threshold_costs.append(run_episode(full_raw_seq, full_scaled_seq, full_pv_seq,
                                       custom_policy_fn=lambda s, t: price_threshold_policy(s, t, threshold=1.0))[0])

# === Save and Plot ===
df_cost = pd.DataFrame({
    "Always0": zero_costs,
    "PriceThreshold": threshold_costs
})
df_cost.to_csv("ppo_vs_price_baselines_30day.csv", index=False)

plt.figure(figsize=(10, 4))
plt.plot(zero_costs, label="Always 0", linewidth=1.2)
plt.plot(threshold_costs, label="Price Threshold", linewidth=1.2)
plt.title("PPO vs Price-Based Strategies (30-Day Episode)")
plt.xlabel("Episode")
plt.ylabel("Total Cost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_vs_price_baselines_30day.png")
plt.show()

print(f"✅ Always 0 Avg Cost:      {np.mean(zero_costs):.2f}")
print(f"✅ Price Threshold Cost:   {np.mean(threshold_costs):.2f}")
