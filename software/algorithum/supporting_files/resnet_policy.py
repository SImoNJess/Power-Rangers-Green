# -*- coding: utf-8 -*-
"""
Created on Sat May 24 12:48:12 2025

@author: zg1223
"""

# === resnet_policy.py ===
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, ReLU, Add, GlobalAveragePooling1D, Dense, LayerNormalization

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv1D(filters, kernel_size, padding='same')
        self.relu1 = ReLU()
        self.conv2 = Conv1D(filters, kernel_size, padding='same')
        self.norm = LayerNormalization(epsilon=1e-6)
        self.relu2 = ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm(inputs + x)  # Residual connection
        return self.relu2(x)

@tf.keras.utils.register_keras_serializable()
class ResNetPolicy(tf.keras.Model):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_proj = Conv1D(64, kernel_size=3, padding='same', activation='relu')
        self.res_block1 = self._make_res_block(64)
        self.res_block2 = self._make_res_block(64)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = Dense(128, activation='relu')
        self.policy_logits = Dense(action_dim)
        self.value = Dense(1)

    def _make_res_block(self, filters):
        return tf.keras.Sequential([
            Conv1D(filters, 3, padding='same', activation='relu'),
            Conv1D(filters, 3, padding='same'),
            LayerNormalization(),
            ReLU(),
        ])

    def call(self, x):
        x = self.input_proj(x)
        x = self.res_block1(x) + x
        x = self.res_block2(x) + x
        x = self.flatten(x)
        x = self.dense(x)
        return self.policy_logits(x), self.value(x)

    def get_config(self):
        config = super().get_config()
        config.update({"state_dim": self.state_dim, "action_dim": self.action_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)