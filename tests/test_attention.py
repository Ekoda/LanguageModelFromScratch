import numpy as np
import pytest
from transformer.attention import MultiHeadAttention, Head

def test_head_attention_shape():
    head = Head(32)

    sequence_length = 10
    embedding_size = 32

    X = np.random.rand(sequence_length, embedding_size)
    weighted_values = head.forward(X)

    assert weighted_values.shape == X.shape

# def test_multihead_attention_forward_output_shape():
#     multihead_attention = MultiHeadAttention(embedding_size=32, n_heads=8)

#     sequence_length = 10
#     embedding_size = 32

#     X = np.random.rand(sequence_length, embedding_size)
#     output = multihead_attention.forward(X)

#     assert output.shape == X.shape