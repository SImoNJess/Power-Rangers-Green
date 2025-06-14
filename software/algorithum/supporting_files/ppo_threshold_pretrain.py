# -*- coding: utf-8 -*-
"""
Supervised Pretraining for PPO Policy using Threshold Strategy
Fixed to generate dataset only once before training
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import MinMaxScaler
from transformer_policy import TransformerPolicy
from ppo_trader import PPOTrader

print("🔧 Starting supervised pretraining for PPO policy...")

# === Constants ===
SEQ_LEN = 60
STATE_DIM = 4
ACTION_SIZE = 17
MAX_STORAGE = 50.0
SAMPLES = 3000
EPOCHS = 1
BATCH = 64
action_space = np.linspace(-40, 40, ACTION_SIZE)

# === Load & preprocess data ===
df = pd.read_csv("all_history_full.txt", sep="\s+")
tick_col = df.iloc[:, 1].values
raw_data = df[["sun", "price_buy", "price_sell", "demand"]].astype(float).values  # jus donnot fucking change this line

pv = np.minimum(raw_data[:, 0], 100.0) * 0.032
smoothed = uniform_filter1d(raw_data, size=10, axis=0)

scaler = MinMaxScaler()
features = scaler.fit_transform(smoothed)
joblib.dump(scaler, "scaler_dqn.pkl")

valid_start = np.where(tick_col == 0)[0]

# === Policy function
def price_threshold_policy(state_seq, t=0):
    last_step_scaled = state_seq[t].reshape(1, -1)
    last_step_unscaled = scaler.inverse_transform(last_step_scaled)[0]

    sell_price = last_step_unscaled[1]
    buy_price = last_step_unscaled[2]

    sell_threshold = 40
    buy_threshold = 30

    if sell_price > sell_threshold:
        return 5-(sell_price-sell_threshold)
    elif buy_price < buy_threshold:
        return 12+(buy_threshold-buy_price)
    else:
        return 8

# === Pre-generate dataset
np.random.seed(42)
X_seq = []
X_storage = []
Y_action = []

count = 0
while count < SAMPLES:
    idx = np.random.choice(valid_start[:-SEQ_LEN])
    if idx + SEQ_LEN + 1 >= len(features):
        continue
    seq = features[idx:idx+SEQ_LEN].astype(np.float32)
    storage = np.random.uniform(0, MAX_STORAGE)
    label = price_threshold_policy(seq)

    X_seq.append(seq)
    X_storage.append([storage / MAX_STORAGE])
    Y_action.append(label)
    count += 1

X_seq = np.array(X_seq, dtype=np.float32)
X_storage = np.array(X_storage, dtype=np.float32)
Y_action = np.array(Y_action, dtype=np.int32)

dataset = tf.data.Dataset.from_tensor_slices(((X_seq, X_storage), Y_action))
dataset = dataset.shuffle(2000).batch(BATCH).prefetch(tf.data.AUTOTUNE)

# === PPO agent (policy and value)
agent = PPOTrader(
    state_dim=(SEQ_LEN, STATE_DIM),
    action_dim=ACTION_SIZE,
    gamma=0.99,
    clip_ratio=0.2,
    lr=3e-4,
    vf_coef=0.5,
    ent_coef=0.01,
)

# === Wrapper model
class LogitWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, training=False):
        logits, _ = self.model(inputs, training=training)
        return logits

policy_model = LogitWrapper(agent.policy)

# === Compile model
steps_per_epoch = SAMPLES // BATCH
policy_model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# === Visualize one sample to verify label and input
(sample_seq, sample_storage), sample_label = next(iter(dataset.take(1)))

# Inverse transform full sequence for detailed inspection
full_unscaled = scaler.inverse_transform(sample_seq[0].numpy())

print("🔎 Full 60-tick sequence (buy/sell prices):")
print(f"{'Tick':>4} | {'Buy Price':>10} | {'Sell Price':>11}")
print("-" * 32)
for i, row in enumerate(full_unscaled):
    buy = row[2]
    sell = row[1]
    label = price_threshold_policy(sample_seq[0].numpy(), t=i)
    print(f"{i:>4} | {buy:>10.2f} | {sell:>11.2f} | {label:>6}")

# Last tick detail
final_tick = full_unscaled[-1]
print("\n🔎 Sample input (last tick of 60):")
print(f"  sun={final_tick[0]:.2f}, buy={final_tick[2]:.2f}, "
      f"sell={final_tick[1]:.2f}, demand={final_tick[3]:.2f}")
#print(f"  Storage level: {sample_storage[0].numpy() * MAX_STORAGE:.2f} J")
#print(f"  Target label (action index): {sample_label.numpy()}")

# === Train model
history = policy_model.fit(
    dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    verbose=1
)

# === Save outputs
agent.policy.save("ppo_trader_policy_pretrained.keras")
pd.DataFrame(history.history).to_csv("ppo_threshold_pretrain_log.csv", index=False)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.title("Pretrain Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title("Pretrain Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("ppo_threshold_pretrain_plot.png")
plt.show()

# === Evaluate: compare prediction vs label
print("\n🔍 Final prediction vs target (first 60 samples):")
preds = policy_model.predict([X_seq[:60], X_storage[:60]], verbose=0)
pred_actions = tf.argmax(preds, axis=1).numpy()
true_actions = Y_action[:60]

print(f"{'Tick':>4} | {'Target':>6} | {'Predicted':>9}")
print("-" * 26)
for t, (label, pred) in enumerate(zip(true_actions, pred_actions)):
    print(f"{t:>4} | {label:>6} | {pred:>9}")

print("✅ Pretraining completed and model saved.")
