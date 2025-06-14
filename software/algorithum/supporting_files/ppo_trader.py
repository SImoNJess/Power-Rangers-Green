# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 21:01:07 2025

@author: zg1223
"""

# === ppo_trader.py ===
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from transformer_policy import TransformerPolicy

class PPOTrader:
    def __init__(self, state_dim, action_dim, gamma=0.99, clip_ratio=0.2, lr=3e-4, vf_coef=0.5, ent_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.policy = TransformerPolicy(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Initialize buffer for storing transitions
        self.buffer = {
            "states": [],
            "actions": [],
            "rewards": [],
            "logps": [],
            "dones": []
        }

    def get_action(self, state_input):
        x_seq = np.expand_dims(state_input[0], axis=0).astype(np.float32)
        x_storage = np.expand_dims(state_input[1], axis=0).astype(np.float32)
        logits, value = self.policy([x_seq, x_storage], training=False)
        action_probs = tf.nn.softmax(logits)
        dist = tf.squeeze(action_probs)
        action = tf.random.categorical(tf.math.log([dist]), 1)[0, 0].numpy()
        return action, tf.math.log(dist[action] + 1e-8).numpy(), value[0, 0].numpy()

    def store(self, state, action, reward, logp, done):
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)
        self.buffer["logps"].append(logp)
        self.buffer["dones"].append(done)

    def compute_advantage(self, rewards, values, dones):
        adv = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * values[t + 1] - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            adv.insert(0, gae)
        return np.array(adv, dtype=np.float32)

    def train(self, batch_size):
        states_seq = np.array([s[0] for s in self.buffer["states"]], dtype=np.float32)
        storage_vals = np.array([s[1] for s in self.buffer["states"]], dtype=np.float32)
        actions = np.array(self.buffer["actions"])
        rewards = np.array(self.buffer["rewards"])
        logps = np.array(self.buffer["logps"])
        dones = np.array(self.buffer["dones"], dtype=np.float32)

        values = []
        for i in range(len(states_seq)):
            v = self.policy([np.expand_dims(states_seq[i], 0), np.expand_dims(storage_vals[i], 0)], training=False)[1].numpy()[0, 0]
            values.append(v)
        values.append(0.0)
        values = np.array(values, dtype=np.float32)
        returns = rewards + self.gamma * values[1:] * (1 - dones)
        advantages = self.compute_advantage(rewards, values, dones)

        loss = self._train([states_seq, storage_vals], actions, logps, returns, advantages)

        # Clear buffer
        self.buffer = {k: [] for k in self.buffer}
        return loss

    def _train(self, states, actions, old_log_probs, returns, advantages):
        actions = tf.convert_to_tensor(actions)
        old_log_probs = tf.convert_to_tensor(old_log_probs)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        x_seq = tf.convert_to_tensor(states[0], dtype=tf.float32)
        x_storage = tf.convert_to_tensor(states[1], dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits, values = self.policy([x_seq, x_storage], training=True)
            values = tf.squeeze(values)

            action_probs = tf.nn.softmax(logits)
            log_probs = tf.math.log(tf.reduce_sum(action_probs * tf.one_hot(actions, self.action_dim), axis=1) + 1e-8)
            ratio = tf.exp(log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            value_loss = tf.reduce_mean(tf.square(returns - values))
            entropy = -tf.reduce_mean(action_probs * tf.math.log(action_probs + 1e-8))

            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        return loss.numpy(), policy_loss.numpy(), value_loss.numpy(), entropy.numpy()
