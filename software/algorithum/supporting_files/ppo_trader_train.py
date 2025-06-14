# -*- coding: utf-8 -*-
"""
PPO Trader with pretrained model and 30-day episodes (continuous storage)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ppo_trader import PPOTrader
from transformer_policy import TransformerPolicy

print("Training PPO Trader using pretrained policy and 30-day window 1000e still use scaled but one devied by 2  ")

# === Constants ===
STATE_SEQ_LEN = 60
STATE_DIM = 4
ACTION_SIZE = 17
MAX_STORAGE = 50
MAX_POWER = 4.0
TICK_DURATION = 5
EPISODES = 1000
BATCH_SIZE = 64
DAYS_PER_EPISODE = 30
TICKS_PER_DAY = 60
EPISODE_TICKS = DAYS_PER_EPISODE * TICKS_PER_DAY

# === Load and preprocess data ===
df = pd.read_csv("all_history_full.txt", sep='\s+')
tick_column = df.iloc[:, 1].values
data = df[["sun", "price_buy", "price_sell", "demand"]].astype(float).values
sun_column = data[:, 0]
pv_raw = np.minimum(sun_column, 100.0) * 0.032

data = uniform_filter1d(data, size=10, axis=0)
scaler = MinMaxScaler()
scaler.fit(data)
features = scaler.transform(data)
joblib.dump(scaler, "scaler_dqn.pkl")

action_space = np.linspace(-40.0, 40.0, ACTION_SIZE)
valid_start_indices = np.where(tick_column == 0)[0]
cost_history, loss_history = [], []

# === PPO Agent with pretrained model ===
agent = PPOTrader(
    state_dim=(STATE_SEQ_LEN, STATE_DIM),
    action_dim=ACTION_SIZE,
    gamma=0.99,
    clip_ratio=0.2,
    lr=3e-4,
    vf_coef=0.5,
    ent_coef=0.01,
)
agent.policy = tf.keras.models.load_model(
    "ppo_trader_policy.keras",
    custom_objects={"TransformerPolicy": TransformerPolicy}
)

# === Training loop ===
for episode in range(EPISODES):
    # Safe selection of valid start points
    valid_candidates = valid_start_indices[valid_start_indices < len(features) - EPISODE_TICKS]
    if len(valid_candidates) == 0:
        raise ValueError("🚫 Not enough data to support a full 30-day episode. Reduce DAYS_PER_EPISODE or collect more data.")
    
    start_idx = np.random.choice(valid_candidates)

    window = features[start_idx:start_idx + EPISODE_TICKS]
    tick_window = tick_column[start_idx:start_idx + EPISODE_TICKS]
    pv_window = pv_raw[start_idx:start_idx + EPISODE_TICKS]

    storage = 35
    total_cost = 0

    for t in range(EPISODE_TICKS - STATE_SEQ_LEN):
        state_seq = window[t:t + STATE_SEQ_LEN]
        storage_input = storage / MAX_STORAGE
        state_input = [state_seq.astype(np.float32), storage_input]

        action, logp, _ = agent.get_action(state_input)
        action_val = action_space[action]

        sun_now, sell_now, buy_now, demand_scaled = window[t]
        sell_now=sell_now/2
        PV_now = pv_window[t]
        demand_now = demand_scaled * MAX_POWER
        demand_energy = demand_now * TICK_DURATION

        cost = 0
        serve_energy = demand_energy
        use_pv = min(PV_now * TICK_DURATION, serve_energy)
        serve_energy -= use_pv
        remaining_pv = PV_now * TICK_DURATION - use_pv

        use_battery = min(storage, serve_energy)
        storage -= use_battery
        serve_energy -= use_battery

        if serve_energy > 0:
            cost += serve_energy * buy_now

        if remaining_pv > 0 and storage < MAX_STORAGE:
            store_from_pv = min(MAX_STORAGE - storage, remaining_pv)
            storage += store_from_pv
            remaining_pv -= store_from_pv

        if remaining_pv > 0:
            cost -= remaining_pv * sell_now

        allowed_action = np.clip(action_val, -storage, MAX_STORAGE - storage)
        storage += allowed_action
        if allowed_action > 0:
            cost += allowed_action * buy_now
        else:
            cost -= -allowed_action * sell_now

        done = (t + 1) == (EPISODE_TICKS - STATE_SEQ_LEN - 1)
        reward = -cost

        agent.store([state_seq.astype(np.float32), storage_input], action, reward, logp, done)
        total_cost += cost

        if done:
            break

    loss = agent.train(BATCH_SIZE)
    cost_history.append(total_cost)
    if loss is not None:
        loss_history.append(loss)
    if(episode%50==0):
        print(f"Episode {episode}, Total Cost: {total_cost:.2f}, Total Loss: {loss[0]:.4f}, Policy: {loss[1]:.4f}, Value: {loss[2]:.4f}, Entropy: {loss[3]:.4f}")

# === Save models and logs ===
agent.policy.save("ppo_trader_policy.keras")
pd.DataFrame({"cost": cost_history}).to_csv("ppo_trader_cost.csv", index=False)
pd.DataFrame({"loss": loss_history}).to_csv("ppo_trader_loss.csv", index=False)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cost_history)
plt.title("PPO Trader Cost per Episode")
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title("PPO Trader Loss per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_trader_training_plot.png")
plt.show()
