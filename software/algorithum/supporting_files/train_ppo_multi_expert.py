# === ppo_multi_expert_train_resnet_eval.py ===
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import uniform_filter1d
from tensorflow.keras import Model
from transformer_multi_expert_policy import TransformerMultiExpertPolicy
from ppo_multi_expert_agent import PPOAgent
from resnet_policy import ResNetPolicy

print("Starting PPO training with 6-task policy...debugged 4")

# === Constants ===
STATE_SEQ_LEN = 60
STATE_DIM = 4
DEFER_DIM = 61
NUM_TASKS = 6
BATCH_SIZE = 32
EPOCHS = 1000
ACTION_SIZE = 17

# === Load pretrained ResNet model ===
resnet_model = tf.keras.models.load_model(
    "trained_models/dqn_resnet_model.keras",
    custom_objects={"ResNetPolicy": ResNetPolicy}
)

# === Load and build scheduler policy ===
policy = TransformerMultiExpertPolicy()
dummy_seq = tf.zeros((1, STATE_SEQ_LEN, STATE_DIM))
dummy_defer = tf.zeros((1, NUM_TASKS, DEFER_DIM))
_ = policy([dummy_seq, dummy_defer])  # Build model
policy.load_weights("trained_models/transformer_scheduler_model.keras")

# === Load and preprocess data ===
scaler = joblib.load("scaler_dqn.pkl")
df = pd.read_csv("all_history_full.txt", delim_whitespace=True)
df = df.tail(len(df) // 12).reset_index(drop=True)

raw_data = df[["sun", "price_buy", "price_sell", "demand"]].astype(np.float32).values
smoothed_data = uniform_filter1d(raw_data, size=5, axis=0, mode='nearest')
features = scaler.transform(smoothed_data).astype(np.float32)

# === ResNet-based cost function ===
def evaluate_cost(schedule, seq_batch, defer_batch):
    schedule = tf.maximum(schedule, 0.0)
    energy = tf.cast(defer_batch[:, :, 0], tf.float32)
    mask = tf.clip_by_value(defer_batch[:, :, 1:], 0.0, 1.0)
    masked_schedule = schedule * mask
    sum_sched = tf.reduce_sum(masked_schedule, axis=-1)
    energy_gap = tf.abs(sum_sched - energy)
    energy_penalty = tf.reduce_sum(energy_gap, axis=-1)

    defer_total = tf.reduce_sum(masked_schedule, axis=1)
    immed_total = tf.cast(seq_batch[:, :, 3], tf.float32)
    full_demand = tf.expand_dims(defer_total + immed_total, -1)

    resnet_input = tf.concat([seq_batch[:, :, :3], full_demand], axis=-1)
    q_values, _ = resnet_model(resnet_input, training=False)
    best_q = tf.reduce_max(q_values, axis=-1)
    base_cost = -best_q

    return base_cost + 5.0 * energy_penalty

# === Deferable load generator ===
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

# === Prepare dataset ===
N = len(features) - STATE_SEQ_LEN
seq_data = np.array([features[i:i+STATE_SEQ_LEN] for i in range(N)], dtype=np.float32)
defer_data = generate_structured_deferable(N)

# === Initialize PPO Agent ===
agent = PPOAgent(policy_class=lambda: policy, state_dim=(60, 4), action_dim=(NUM_TASKS, 60))

loss_history = []
cost_history = []

# === Training loop ===
for epoch in range(EPOCHS):
    idx = np.random.choice(len(seq_data), BATCH_SIZE)
    seq_batch = tf.convert_to_tensor(seq_data[idx], dtype=tf.float32)
    defer_batch = tf.convert_to_tensor(defer_data[idx], dtype=tf.float32)

    with tf.GradientTape() as tape:
        schedule = agent.old_policy([seq_batch, defer_batch])
        cost = evaluate_cost(schedule, seq_batch, defer_batch)
        reward = -cost
        loss = agent.compute_loss([seq_batch, defer_batch], schedule, reward)

    grads = tape.gradient(loss, agent.policy.trainable_variables)
    agent.optimizer.apply_gradients(zip(grads, agent.policy.trainable_variables))
    
    loss_history.append(loss.numpy())
    cost_history.append(tf.reduce_mean(cost).numpy())
    
    if epoch % 50 == 0:
        agent.update_old_policy()
        print(f"Epoch {epoch}: Avg Cost = {cost_history[-1]:.3f}, Loss = {loss_history[-1]:.3f}")

# === Save model ===
policy.save("trained_models/ppo_multi_expert_policy_final.keras")
print("✅ Model saved to trained_models/ppo_multi_expert_policy_final.keras")

# === Plot training progress ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label="Loss")
plt.title("PPO Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cost_history, label="Cost", color="orange")
plt.title("Average Cost Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("ppo_multi_expert_training_curve.png")
plt.show()
