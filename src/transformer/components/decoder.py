import numpy as np
from src.utils.type_utils import Matrix
from src.transformer.components.attention import MultiHeadAttention
from src.neural_net.network import FeedForwardNetwork, NeuralComponent
from src.transformer.components.layer_norm import LayerNorm
from src.utils.math_utils import add


class Decoder(NeuralComponent):
    def __init__(self, embedding_size: int, n_attention_heads: int = 8):
        self.embedding_size: int = embedding_size
        self.n_heads: int = n_attention_heads
        self.masked_attention = MultiHeadAttention(embedding_size, n_attention_heads, masked=True)
        self.masked_attention_norm = LayerNorm(embedding_size)
        self.feed_forward_network = FeedForwardNetwork(embedding_size, embedding_size)
        self.feed_forward_norm = LayerNorm(embedding_size)

    def parameters(self):
        parameters = []
        parameters.extend(self.masked_attention.parameters())
        parameters.extend(self.masked_attention_norm.parameters())
        parameters.extend(self.feed_forward_network.parameters())
        parameters.extend(self.feed_forward_norm.parameters())
        return parameters

    def forward(self, X: Matrix) -> Matrix:
        attention_layer = self.masked_attention.forward(X)
        attention_add_and_norm = self.masked_attention_norm.forward(add(X, attention_layer))
        feed_forward = [self.feed_forward_network.forward(embedding) for embedding in X]
        feed_forward_add_and_norm = self.feed_forward_norm.forward(add(attention_add_and_norm, feed_forward))
        return feed_forward_add_and_norm
