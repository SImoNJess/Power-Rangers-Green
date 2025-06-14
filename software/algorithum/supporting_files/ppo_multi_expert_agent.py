import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class PPOAgent:
    def __init__(self, policy_class, state_dim, action_dim, clip_ratio=0.2, lr=3e-4):
        # === 创建主策略网络 ===
        self.policy = policy_class()
        self.policy.build(input_shape=[(None, 60, 4), (None, 10, 61)])

        # === 创建旧策略网络并同步参数 ===
        self.old_policy = policy_class()
        self.old_policy.build(input_shape=[(None, 60, 4), (None, 10, 61)])
        self.old_policy.set_weights(self.policy.get_weights())

        # === PPO 参数 ===
        self.optimizer = Adam(learning_rate=lr)
        self.clip_ratio = clip_ratio

    def compute_loss(self, inputs, actions, rewards):
        # 获取当前策略和旧策略的 schedule 输出
        logits = self.policy(inputs)
        old_logits = self.old_policy(inputs)

        # 简化版 ratio（MSE proxy）
        numerator = tf.reduce_sum((old_logits - actions)**2, axis=[1, 2])
        denominator = tf.reduce_sum((logits - actions)**2, axis=[1, 2]) + 1e-8
        ratio = numerator / denominator
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        # Advantage = reward - baseline
        advantages = rewards - tf.reduce_mean(rewards)

        # PPO clipped surrogate loss
        loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        return loss
    def update_old_policy(self):
        self.old_policy.set_weights(self.policy.get_weights())