import numpy as np
import pytest
from src.transformer.preprocessing.tokenization import build_vocab
from src.transformer.components.embedding import generate_embeddings


def test_build_vocab():
    # Test with a simple list of tokens
    tokens = ["apple", "banana", "cherry", "banana", "apple"]
    expected_vocab = {"apple": 0, "banana": 1, "cherry": 2}
    assert build_vocab(tokens) == expected_vocab

    # Test with an empty list of tokens
    tokens = []
    expected_vocab = {}
    assert build_vocab(tokens) == expected_vocab

    # Test with a list of one token
    tokens = ["apple"]
    expected_vocab = {"apple": 0}
    assert build_vocab(tokens) == expected_vocab

def test_generate_embeddings():
    # Test with a small vocabulary and embedding size
    vocab_size = 3
    embedding_size = 2
    embeddings = generate_embeddings(vocab_size, embedding_size)
    assert embeddings.shape == (vocab_size, embedding_size)
    assert np.all(embeddings >= 0) and np.all(embeddings <= 0.01)

    # Test with a larger vocabulary and embedding size
    vocab_size = 100
    embedding_size = 50
    embeddings = generate_embeddings(vocab_size, embedding_size)
    assert embeddings.shape == (vocab_size, embedding_size)
    assert np.all(embeddings >= 0) and np.all(embeddings <= 0.01)

    # Test with a vocabulary size of 1
    vocab_size = 1
    embedding_size = 1
    embeddings = generate_embeddings(vocab_size, embedding_size)
    assert embeddings.shape == (vocab_size, embedding_size)
    assert np.all(embeddings >= 0) and np.all(embeddings <= 0.01)