import numpy as np
from transformer.attention import MultiHeadAttention
from transformer.neural_network import NeuronLayer, FeedForwardNetwork
from transformer.layer_norm import LayerNorm


class Decoder:
    def __init__(self, embedding_size: int, n_attention_heads: int = 8):
        self.embedding_size: int = embedding_size
        self.n_heads: int = n_attention_heads
        self.masked_attention = MultiHeadAttention(embedding_size, n_attention_heads, masked=True)
        self.masked_attention_norm = LayerNorm(embedding_size)
        self.feed_forward_network = FeedForwardNetwork(embedding_size)
        self.feed_forward_norm = LayerNorm(embedding_size)

    def forward(self, X: np.ndarray) -> np.ndarray:
        attention_layer = self.masked_attention.forward(X)
        attention_add_and_norm = self.masked_attention_norm.forward(X + attention_layer)
        feed_forward = np.array([self.feed_forward_network.forward(embedding) for embedding in X])
        feed_forward_add_and_norm = self.feed_forward_norm.forward(attention_add_and_norm + feed_forward)
        return feed_forward_add_and_norm
