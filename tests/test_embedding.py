import numpy as np
import pytest
from src.preprocessing.tokenization import build_vocab
from src.transformer.components.embedding import generate_embeddings
from src.utils.math_utils import get_shape


def test_build_vocab():
    tokens = ["apple", "banana", "cherry", "banana", "apple"]
    expected_vocab = {"apple": 0, "banana": 1, "cherry": 2}
    assert build_vocab(tokens) == expected_vocab

    tokens = []
    expected_vocab = {}
    assert build_vocab(tokens) == expected_vocab

    tokens = ["apple"]
    expected_vocab = {"apple": 0}
    assert build_vocab(tokens) == expected_vocab

def test_generate_embeddings():
    vocab_size = 3
    embedding_size = 2
    embeddings = generate_embeddings(vocab_size, embedding_size)
    assert get_shape(embeddings) == (vocab_size, embedding_size)

    vocab_size = 100
    embedding_size = 50
    embeddings = generate_embeddings(vocab_size, embedding_size)
    assert get_shape(embeddings) == (vocab_size, embedding_size)

    vocab_size = 1
    embedding_size = 1
    embeddings = generate_embeddings(vocab_size, embedding_size)
    assert get_shape(embeddings) == (vocab_size, embedding_size)
