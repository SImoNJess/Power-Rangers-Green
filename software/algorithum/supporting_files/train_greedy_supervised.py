import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

from transformer_multi_expert_policy import TransformerMultiExpertPolicy
print("fuck the ic and project 60 with 6 tasks from 10 to 100e")

# === Load & preprocess ===
def load_tick_windows(path="all_history_full.txt", smooth=5, scaler_path="scaler_dqn.pkl"):
    df = pd.read_csv(path, delim_whitespace=True)
    data = df[["sun", "price_buy", "price_sell", "demand"]].astype(float).values
    data = uniform_filter1d(data, size=smooth, axis=0, mode="nearest")
    scaler = joblib.load(scaler_path)
    data = scaler.transform(data)
    return np.stack([data[i:i + 60] for i in range(data.shape[0] - 60 + 1)], axis=0)

# === Greedy scheduling ===
def greedy_schedule(PV, D_fixed, P_buy, P_sell, tasks):
    T = 60
    D_def = np.zeros(T)
    schedule = []
    for E_i, t_start, t_end in tasks:
        time_range = list(range(t_start, t_end))
        tick_scores = []
        for t in time_range:
            net_surplus = PV[t] - D_fixed[t]
            score = -net_surplus if net_surplus > 0 else P_buy[t]
            tick_scores.append((t, score))
        tick_scores.sort(key=lambda x: x[1])
        remaining = E_i
        plan = []
        for t, _ in tick_scores:
            if remaining <= 0:
                break
            net_surplus = PV[t] - D_fixed[t]
            available_power = 4.0 if net_surplus > 0 else 2.5
            max_energy = max(0, 20.0 - (D_fixed[t] + D_def[t]) * 5)
            alloc = min(available_power * 5, remaining, max_energy)
            D_def[t] += alloc / 5
            remaining -= alloc
            plan.append((t, alloc))
        schedule.append(plan)
    return D_def * 5, schedule

# === Label generation ===
def generate_dataset(X_seq, task_count=6):
    X_tasks, Y_sched = [], []
    for x in X_seq:
        PV = np.minimum(x[:, 0], 1.0) * 3.6
        D_fixed = x[:, 3] * 4.0
        P_buy = x[:, 1]
        P_sell = x[:, 2]
        tasks = []
        for _ in range(task_count):
            E = np.random.uniform(10, 100)
            t_start = np.random.randint(0, 50)
            t_end = np.random.randint(t_start + 5, min(t_start + 20, 60))
            tasks.append((E, t_start, t_end))
        _, sched = greedy_schedule(PV, D_fixed, P_buy, P_sell, tasks)
        x_task = np.zeros((task_count, 61))
        y_task = np.zeros((task_count, 60))
        for i, (E_i, t_start, t_end) in enumerate(tasks):
            x_task[i, 0] = E_i
            x_task[i, t_start + 1:t_end + 1] = 1
            for t, a in sched[i]:
                y_task[i, t] = a
        X_tasks.append(x_task)
        Y_sched.append(y_task)
    return np.array(X_tasks), np.array(Y_sched)

# === Train & Test ===
def main():
    print("🔄 Loading and scaling data...")
    X_seq = load_tick_windows("all_history_full.txt", scaler_path="scaler_dqn.pkl")

    print("🎯 Generating greedy labels...")
    X_tasks, Y_sched = generate_dataset(X_seq)

    print("🧠 Building and training model...")
    model = TransformerMultiExpertPolicy()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
    model.fit([X_seq, X_tasks], Y_sched, batch_size=128, epochs=120, validation_split=0.1)

    print("🔍 Running inference...")
    Y_pred = model.predict([X_seq, X_tasks], batch_size=64)

    # === Visualize and Compare ===
    idx = 250
    x_task = X_tasks[idx]
    y_true = Y_sched[idx]
    y_pred = Y_pred[idx]

    unmet_tasks = []
    plt.figure(figsize=(10, 6))
    for i in range(6):
        E_required = x_task[i, 0]
        mask = x_task[i, 1:]
        pred_energy = np.sum(y_pred[i] * mask)
        true_energy = np.sum(y_true[i])

        if pred_energy < 0.95 * E_required:
            unmet_tasks.append(i)

        plt.plot(y_true[i], linestyle='dashed', label=f"True Task {i}")
        plt.plot(y_pred[i], label=f"Pred Task {i} (need {E_required:.1f}, got {pred_energy:.1f})")

    plt.title("Predicted vs True Schedule (Sample 250)")
    plt.xlabel("Tick")
    plt.ylabel("Energy (Joules)")
    plt.grid(True)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("predicted_schedule_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "transformer_scheduler_model.keras"))
    print("✅ Model saved to transformer_scheduler_model.keras")

    if unmet_tasks:
        print("⚠️ Tasks that failed to meet energy requirement:", unmet_tasks)
    else:
        print("✅ All predicted tasks met energy requirements")

    print("📐 Prediction shape:", Y_pred.shape)

main()
