import numpy as np
from transformer.attention import MultiHeadAttention
from transformer.neural_network import LayerNorm, NeuronLayer

class Decoder:
    def __init__(self, embedding_size: int, n_attention_heads: int = 8):
        self.embedding_size: int = embedding_size
        self.n_heads: int = n_attention_heads
        self.masked_multi_head_attention = MultiHeadAttention(embedding_size, n_attention_heads, masked=True)
        self.masked_attention_norm = LayerNorm(embedding_size)
        self.ReluLayer = NeuronLayer(embedding_size, embedding_size, activation='relu')
        self.linear_layer = NeuronLayer(embedding_size, embedding_size, activation='linear')
        self.feed_forward_norm = LayerNorm(embedding_size)


    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        relu_layer = np.array([self.ReluLayer.forward(x) for x in X])
        linear_layer = np.array([self.linear_layer.forward(r) for r in relu_layer])
        return linear_layer

    def forward(self, X: np.ndarray) -> np.ndarray:
        attention_layer = self.masked_multi_head_attention.forward(X)
        attention_add_and_norm = self.masked_attention_norm.forward(X + attention_layer)
        feed_forward = self.feed_forward(attention_add_and_norm)
        feed_forward_add_and_norm = self.feed_forward_norm.forward(attention_add_and_norm + feed_forward)
        return feed_forward_add_and_norm