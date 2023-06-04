import numpy as np
from transformer.neural_network import NeuronLayer
from utils.math_utils import softmax

class MultiHeadAttention:
    def __init__ (self, embedding_size: int, n_heads: int = 8):
        self.embedding_size: int = embedding_size
        self.n_heads: int = n_heads
        self.heads = [Head(embedding_size // n_heads) for _ in range(n_heads)]
        self.linear_layer = NeuronLayer(embedding_size, embedding_size, activation='linear', include_bias=False)

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass


class Head:
    def __init__ (self, size: int):
        self.size: int = size
        self.query_layer = NeuronLayer(size, size, activation='linear', include_bias=False)
        self.key_layer = NeuronLayer(size, size, activation='linear', include_bias=False)
        self.value_layer = NeuronLayer(size, size, activation='linear', include_bias=False)


    def forward(self, X: np.ndarray) -> np.ndarray:
        queries = np.array([self.query_layer.forward(embedding) for embedding in X]) # T, C 
        keys = np.array([self.key_layer.forward(embedding) for embedding in X]) # T, C
        values = np.array([self.value_layer.forward(embedding) for embedding in X]) # T, C
        
        scores = softmax(np.matmul(queries, keys.T) / np.sqrt(self.size), axis=-1) # T, T
        weighted_values = np.matmul(scores, values) # T, C

        return weighted_values