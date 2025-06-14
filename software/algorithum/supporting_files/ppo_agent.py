# === ppo_agent.py ===
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class PPOAgent:
    def __init__(self, policy_class, state_dim, action_dim, clip_ratio=0.2, lr=3e-4, gamma=0.99, lambd=0.97):
        self.policy = policy_class(state_dim, action_dim)
        self.old_policy = policy_class(state_dim, action_dim)
        self.old_policy.set_weights(self.policy.get_weights())

        self.optimizer = Adam(learning_rate=lr)
        self.gamma = gamma
        self.lambd = lambd
        self.clip_ratio = clip_ratio

    def select_action(self, state):
        prob, value = self.policy(state)
        action = tf.random.categorical(tf.math.log(prob), 1)[0, 0]
        return int(action.numpy()), float(value.numpy())

    def train(self, states, actions, rewards, dones, values):
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        returns, advantages = self._compute_gae(rewards, dones, values)

        actions = np.array(actions, dtype=np.int32)
        returns = returns.astype(np.float32)
        advantages = advantages.astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((states, actions, returns, advantages)).shuffle(1024).batch(32)

        for state_batch, action_batch, return_batch, adv_batch in dataset:
            with tf.GradientTape() as tape:
                probs, values = self.policy(state_batch)
                old_probs, _ = self.old_policy(state_batch)

                values = tf.squeeze(values)
                old_probs = tf.stop_gradient(old_probs)

                action_masks = tf.one_hot(action_batch, probs.shape[-1])
                prob_act = tf.reduce_sum(probs * action_masks, axis=-1)
                old_prob_act = tf.reduce_sum(old_probs * action_masks, axis=-1)
                ratio = prob_act / (old_prob_act + 1e-10)

                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv_batch, clipped_ratio * adv_batch))

                value_loss = tf.reduce_mean((return_batch - values) ** 2)
                entropy = -tf.reduce_mean(probs * tf.math.log(probs + 1e-10))

                loss = policy_loss + 1 * value_loss - 0.01 * entropy

            grads = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        self.old_policy.set_weights(self.policy.get_weights())

    def _compute_gae(self, rewards, dones, values):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambd * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        return returns, advantages
