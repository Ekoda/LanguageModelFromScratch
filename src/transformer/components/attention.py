import numpy as np

from src.neural_net.network import NeuronLayer, NeuralComponent
from src.utils.attention_utils import mask_attention_scores
from src.utils.math_utils import softmax, transpose, matmul, apply_elementwise
from src.utils.type_utils import Matrix


class MultiHeadAttention(NeuralComponent):
    def __init__ (self, embedding_size: int, n_heads: int = 8, masked: bool = False):
        self.embedding_size: int = embedding_size
        self.n_heads: int = n_heads
        self.masked: bool = masked
        self.heads = [Head(embedding_size, embedding_size // n_heads, masked=masked) for _ in range(n_heads)]
        self.linear_layer = NeuronLayer(embedding_size, embedding_size, activation='linear', include_bias=False)

    def parameters(self):
        parameters = []
        for head in self.heads:
            parameters.extend(head.parameters())
        parameters.extend(self.linear_layer.parameters())
        return parameters

    def forward(self, X: Matrix) -> Matrix:
        head_outputs = [head.forward(X) for head in self.heads]
        concat = [x + y for x, y in zip(*head_outputs)] if self.n_heads > 1 else head_outputs[0]
        linear_transformation = [self.linear_layer.forward(embedding) for embedding in concat]
        return linear_transformation


class Head(NeuralComponent):
    def __init__ (self, embedding_size: int, head_size: int, masked: bool = False):
        self.head_size: int = head_size
        self.masked: bool = masked
        self.query_layer = NeuronLayer(embedding_size, head_size, activation='linear', include_bias=False)
        self.key_layer = NeuronLayer(embedding_size, head_size, activation='linear', include_bias=False)
        self.value_layer = NeuronLayer(embedding_size, head_size, activation='linear', include_bias=False)

    def parameters(self):
        parameters = []
        parameters.extend(self.query_layer.parameters())
        parameters.extend(self.key_layer.parameters())
        parameters.extend(self.value_layer.parameters())
        return parameters

    def forward(self, X: Matrix) -> Matrix:
        queries = [self.query_layer.forward(embedding) for embedding in X] # T, C 
        keys = [self.key_layer.forward(embedding) for embedding in X] # T, C
        values = [self.value_layer.forward(embedding) for embedding in X] # T, C
        raw_scores = matmul(queries, transpose(keys)) # T, T
        scores = apply_elementwise(raw_scores, lambda x: x / np.sqrt(self.head_size)) # T, T - normalized scores
        if self.masked:
            scores = mask_attention_scores(scores) # T, T    
        softmax_scores = softmax(scores) # T, T
        weighted_values = matmul(softmax_scores, values) # T, C
        return weighted_values
