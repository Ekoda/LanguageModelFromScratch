import pytest
import numpy as np
from src.transformer.decoder import Decoder


def test_forward():
    decoder = Decoder(embedding_size=512, n_attention_heads=8)
    X = np.random.rand(8, 512)
    output = decoder.forward(X)
    assert output.shape == X.shape