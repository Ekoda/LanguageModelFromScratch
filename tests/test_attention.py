import pytest
import numpy as np
from src.transformer.components.attention import MultiHeadAttention, Head
from src.utils.math_utils import softmax, get_shape, matmul, transpose
from src.utils.attention_utils import mask_attention_scores
from src.utils.data_utils import mock_matrix


def test_head_attention_shape():
    head = Head(32)
    X = mock_matrix(10, 32)
    weighted_values = head.forward(X)
    assert get_shape(weighted_values) == get_shape(X)

def test_head_scores():
    head = Head(64)
    X = mock_matrix(5, 64)
    queries = [head.query_layer.forward(embedding) for embedding in X]
    keys = [head.key_layer.forward(embedding) for embedding in X]
    scores = softmax(matmul(queries, transpose(keys)) / np.sqrt(head.size))
    assert get_shape(scores) == (5, 5), "Scores shape does not match expected shape."

def test_multihead_attention_forward_output_shape():
    sequence_length, embedding_size = 8, 32
    multihead_attention = MultiHeadAttention(embedding_size=embedding_size, n_heads=2)
    X = mock_matrix(sequence_length, embedding_size)
    output = multihead_attention.forward(X)
    assert get_shape(output) == get_shape(X), "Output shape does not match input shape."

def test_mask_attention_scores():
    scores = mock_matrix(3, 3)
    masked_scores = mask_attention_scores(scores)
    softmax_scores = softmax(masked_scores)
    assert all([abs(a - b) < 1e-10 for a, b in zip(softmax_scores[0][1:], [0., 0.])])
    assert all([abs(a - b) < 1e-10 for a, b in zip(softmax_scores[1][2:], [0.])])
    assert softmax_scores[2][3:] == []

    scores = mock_matrix(2,2)
    masked_scores = mask_attention_scores(scores)
    softmax_scores = softmax(masked_scores)
    assert all([abs(a - b) < 1e-10 for a, b in zip(softmax_scores[0][1:], [0.])])
    assert softmax_scores[1][2:] == []

    scores = mock_matrix(1,1)
    masked_scores = mask_attention_scores(scores)
    softmax_scores = softmax(masked_scores)
    assert softmax_scores[0][1:] == []