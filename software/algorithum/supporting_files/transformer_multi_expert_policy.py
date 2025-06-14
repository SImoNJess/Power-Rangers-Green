import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, MultiHeadAttention, Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ReLU
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, ff_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(hidden_dim),
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def call(self, x, training=False):
        attn = self.att(x, x)
        x = self.norm1(x + self.drop1(attn, training=training))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.drop2(ffn_out, training=training))

class TransformerExpert(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, ff_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.cross_att = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(hidden_dim)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def call(self, defer_encoded, seq_encoded, training=False):
        attn_output = self.cross_att(defer_encoded, seq_encoded, seq_encoded)
        out1 = self.norm1(defer_encoded + self.drop1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.norm2(out1 + self.drop2(ffn_output, training=training))

class StackedTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, hidden_dim, ff_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.blocks = [TransformerEncoder(hidden_dim, ff_dim, num_heads, dropout_rate) for _ in range(num_layers)]

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        return x

class DeepTransformerExpert(tf.keras.layers.Layer):
    def __init__(self, depth, hidden_dim, ff_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.blocks = [TransformerExpert(hidden_dim, ff_dim, num_heads, dropout_rate) for _ in range(depth)]

    def call(self, defer_encoded, seq_encoded, training=False):
        x = defer_encoded
        for block in self.blocks:
            x = block(x, seq_encoded, training=training)
        return x
class ResidualDenseBlock(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.dense1 = Dense(units, activation='relu')
        self.dense2 = Dense(units)
        self.norm = LayerNormalization(epsilon=1e-6)
        self.relu = ReLU()
        self.skip_proj = None  # dynamic skip projection

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim != self.units:
            self.skip_proj = Dense(self.units)  # align dimension
        super().build(input_shape)

    def call(self, x):
        skip = x
        x = self.dense1(x)
        x = self.dense2(x)
        if self.skip_proj is not None:
            skip = self.skip_proj(skip)
        x = self.norm(x + skip)
        return self.relu(x)
class TransformerMultiExpertPolicy(tf.keras.Model):
    def __init__(self,
                 hidden_dim=64,
                 ff_dim=256,
                 num_heads=8,
                 num_experts=4,
                 encoder_layers=3,
                 expert_depth=2,
                 dropout_rate=0.1):
        super().__init__()
        self.seq_proj = Dense(hidden_dim)
        self.defer_proj = Dense(hidden_dim)

        self.seq_encoder = StackedTransformerEncoder(encoder_layers, hidden_dim, ff_dim, num_heads, dropout_rate)
        self.defer_encoder = StackedTransformerEncoder(encoder_layers, hidden_dim, ff_dim, num_heads, dropout_rate)
        self.seq_pos_embed = tf.keras.layers.Embedding(input_dim=60, output_dim=hidden_dim)
        self.defer_pos_embed = tf.keras.layers.Embedding(input_dim=10, output_dim=hidden_dim)
        self.experts = [
            DeepTransformerExpert(expert_depth, hidden_dim, ff_dim, num_heads, dropout_rate)
            for _ in range(num_experts)
        ]

        self.concat = Concatenate()
        self.output_head = tf.keras.Sequential([
            ResidualDenseBlock(1024),
            ResidualDenseBlock(1024),
            ResidualDenseBlock(512),
            Dense(60)  # Final output: (B, 10, 60)
            ])

    def call(self, inputs, training=False):
        seq_input, defer_input = inputs  # (B, 60, 4), (B, 10, 61)
    
        x_seq = self.seq_proj(seq_input)      # (B, 60, hidden_dim)
        x_defer = self.defer_proj(defer_input)  # (B, 10, hidden_dim)
    
        # === Add positional encoding to sequence input ===
        seq_pos = tf.expand_dims(tf.range(tf.shape(seq_input)[1]), 0)        # (1, 60)
        seq_pos = tf.tile(seq_pos, [tf.shape(seq_input)[0], 1])              # (B, 60)
        x_seq += self.seq_pos_embed(seq_pos)  # self.seq_pos_embed = Embedding(60, hidden_dim)
    
        # === Optional: Add positional encoding to defer input (if time matters for defer range) ===
        defer_pos = tf.expand_dims(tf.range(tf.shape(defer_input)[1]), 0)    # (1, 10)
        defer_pos = tf.tile(defer_pos, [tf.shape(defer_input)[0], 1])        # (B, 10)
        x_defer += self.defer_pos_embed(defer_pos)  # self.defer_pos_embed = Embedding(10, hidden_dim)
    
        # === Encode ===
        x_seq_encoded = self.seq_encoder(x_seq, training=training)
        x_defer_encoded = self.defer_encoder(x_defer, training=training)
    
        expert_outputs = [expert(x_defer_encoded, x_seq_encoded, training=training) for expert in self.experts]
        fused = self.concat(expert_outputs)
        return self.output_head(fused)


    def get_config(self):
        return {
            "hidden_dim": 256,
            "ff_dim": 1024,
            "num_heads": 8,
            "num_experts": 4,
            "encoder_layers": 4,
            "expert_depth": 3,
            "dropout_rate": 0.1,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
