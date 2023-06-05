import numpy as np
import pytest
from transformer.attention import MultiHeadAttention, Head
from utils.math_utils import softmax


def test_head_attention_shape():
    head = Head(32)
    X = np.random.rand(10, 32)
    weighted_values = head.forward(X)
    assert weighted_values.shape == X.shape

def test_head_scores():
    head = Head(64)
    X = np.random.randn(5, 64)
    queries = np.array([head.query_layer.forward(embedding) for embedding in X])
    keys = np.array([head.key_layer.forward(embedding) for embedding in X])
    scores = softmax(np.matmul(queries, keys.T) / np.sqrt(head.size), axis=-1)
    assert scores.shape == (5, 5), "Scores shape does not match expected shape."

def test_multihead_attention_forward_output_shape():
    sequence_length, embedding_size = 8, 32
    multihead_attention = MultiHeadAttention(embedding_size=embedding_size, n_heads=2)
    X = np.random.rand(sequence_length, embedding_size)
    output = multihead_attention.forward(X)
    assert output.shape == X.shape, "Output shape does not match input shape."
