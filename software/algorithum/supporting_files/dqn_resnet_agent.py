# === dqn_resnet_agent.py ===
import numpy as np
import tensorflow as tf
from resnet_policy import ResNetPolicy
from collections import deque
import random

class DQNResNetAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-4, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, memory_size=3000, batch_size=128):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.model = ResNetPolicy(state_dim, action_dim)
        self.target_model = ResNetPolicy(state_dim, action_dim)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_values, _ = self.model(tf.convert_to_tensor(state, dtype=tf.float32))
        return int(tf.argmax(q_values[0]).numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([m[0] for m in minibatch], dtype=np.float32)
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch], dtype=np.float32)
        next_states = np.array([m[3] for m in minibatch], dtype=np.float32)
        dones = np.array([m[4] for m in minibatch], dtype=np.float32)

        next_qs, _ = self.target_model(tf.convert_to_tensor(next_states))
        max_next_q = tf.reduce_max(next_qs, axis=1)
        target_q = rewards + self.gamma * max_next_q.numpy() * (1 - dones)

        with tf.GradientTape() as tape:
            qs, _ = self.model(tf.convert_to_tensor(states))
            one_hot_actions = tf.one_hot(actions, self.action_dim)
            pred_q = tf.reduce_sum(qs * one_hot_actions, axis=1)
            loss = self.loss_fn(target_q, pred_q)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.numpy()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
