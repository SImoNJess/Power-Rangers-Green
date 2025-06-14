import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Embedding, ReLU, Add, Conv1D

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class DenseResidualBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense1 = Dense(hidden_dim, activation='relu')
        self.dense2 = Dense(hidden_dim)
        self.norm = LayerNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = Add()([inputs, x])
        x = self.norm(x)
        return self.relu(x)

class TransformerPolicy(Model):
    def __init__(self, input_shape, num_actions, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = input_shape
        self.num_actions = num_actions

        embed_dim = 64  # Reduced
        num_heads = 4   # Reduced
        ff_dim = 128    # Reduced
        hidden_dim = 128  # Reduced

        self.conv1 = Conv1D(filters=embed_dim, kernel_size=3, padding='same', activation='relu')

        self.embedding = Dense(embed_dim)
        self.pos_embed = Embedding(input_dim=input_shape[0], output_dim=embed_dim)

        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(2)
        ]

        self.pooling = GlobalAveragePooling1D()
        self.pre_dense = Dense(hidden_dim)
        self.dense_res1 = DenseResidualBlock(hidden_dim)
        self.final_dense = Dense(64, activation='relu')

        self.policy_logits = Dense(num_actions)
        self.value_output = Dense(1)

        self.build([(None, *input_shape), (None,)])

    @tf.function
    def call(self, inputs, training=False):
        x_seq = inputs[0]
        x_storage = inputs[1]

        x = self.conv1(x_seq)
        x = self.embedding(x)

        positions = tf.range(tf.shape(x_seq)[1])[tf.newaxis, :]
        x += self.pos_embed(positions)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.pooling(x)

        if len(x_storage.shape) == 1:
            x_storage = tf.expand_dims(x_storage, axis=-1)
        x = tf.concat([x, x_storage], axis=-1)

        x = self.pre_dense(x)
        x = self.dense_res1(x)
        x = self.final_dense(x)

        return self.policy_logits(x), self.value_output(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_,
            "num_actions": self.num_actions
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
