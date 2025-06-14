# -*- coding: utf-8 -*-
"""
Created on Sat May 31 10:38:38 2025
@author: zg1223
"""

# === ppo_multi_expert_test_eval.py ===
import numpy as np
import pandas as pd
import joblib
from scipy.ndimage import uniform_filter1d
import tensorflow as tf
from transformer_multi_expert_policy import TransformerMultiExpertPolicy

# === Constants ===
STATE_SEQ_LEN = 60
STATE_DIM = 4
DEFER_DIM = 61
NUM_TASKS = 6
BATCH_SIZE = 32

# === Load and build model ===
policy = TransformerMultiExpertPolicy()
dummy_seq = tf.zeros((1, STATE_SEQ_LEN, STATE_DIM))
dummy_defer = tf.zeros((1, NUM_TASKS, DEFER_DIM))
_ = policy([dummy_seq, dummy_defer])
policy.load_weights("trained_models/ppo_multi_expert_policy_final.keras")
print("✅ Loaded trained model.")

# === Load and preprocess input ===
scaler = joblib.load("scaler_dqn.pkl")
df = pd.read_csv("all_history_full.txt", delim_whitespace=True)
df = df.tail(len(df) // 12).reset_index(drop=True)

raw_data = df[["sun", "price_buy", "price_sell", "demand"]].astype(np.float32).values
smoothed_data = uniform_filter1d(raw_data, size=5, axis=0, mode='nearest')
features = scaler.transform(smoothed_data).astype(np.float32)

# === Generate test data ===
def generate_structured_deferable(batch_size, num_tasks=NUM_TASKS):
    defer_input = np.zeros((batch_size, num_tasks, DEFER_DIM), dtype=np.float32)
    for b in range(batch_size):
        for i in range(num_tasks):
            energy = np.random.uniform(5, 50)
            start = np.random.randint(0, 50)
            end = np.random.randint(start + 5, 60)
            mask = np.zeros(60)
            mask[start:end] = 1.0
            defer_input[b, i, 0] = energy
            defer_input[b, i, 1:] = mask
    return defer_input

N = len(features) - STATE_SEQ_LEN
seq_data = np.array([features[i:i+STATE_SEQ_LEN] for i in range(N)], dtype=np.float32)
idx = np.random.choice(N, BATCH_SIZE)
seq_batch = tf.convert_to_tensor(seq_data[idx], dtype=tf.float32)
defer_batch = tf.convert_to_tensor(generate_structured_deferable(BATCH_SIZE), dtype=tf.float32)

# === Inference ===
output = policy([seq_batch, defer_batch])
output = tf.maximum(output, 0.0)

# === Energy evaluation ===
required_energy = defer_batch[:, :, 0]
mask = defer_batch[:, :, 1:]
scheduled_energy = tf.reduce_sum(output * mask, axis=-1)
error = tf.abs(scheduled_energy - required_energy)
avg_error = tf.reduce_mean(error).numpy()

# === Percentage error evaluation ===
percentage_error = tf.where(
    required_energy > 0,
    (error / required_energy) * 100.0,
    tf.zeros_like(error)
)
avg_percentage_error = tf.reduce_mean(percentage_error).numpy()

# === Print summary ===
print(f"\n✅ Average absolute energy error per task: {avg_error:.2f} J")
print(f"✅ Average percentage energy error: {avg_percentage_error:.2f} %\n")

# === Print details for a few samples ===
for i in range(min(5, BATCH_SIZE)):
    print(f"--- Sample {i+1} ---")
    for j in range(NUM_TASKS):
        req = required_energy[i, j].numpy()
        sched = scheduled_energy[i, j].numpy()
        err = error[i, j].numpy()
        err_pct = percentage_error[i, j].numpy()
        print(f"Task {j+1}: Required={req:.1f}J | Scheduled={sched:.1f}J | "
              f"Error={err:.2f}J | Error%={err_pct:.2f}%")
