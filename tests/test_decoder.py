import pytest
import numpy as np
from src.transformer.components.decoder import Decoder
from src.utils.data_utils import mock_matrix
from src.utils.math_utils import get_shape


def test_forward():
    decoder = Decoder(embedding_size=16, n_attention_heads=2)
    X = mock_matrix(8, 16)
    output = decoder.forward(X)
    assert get_shape(output) == get_shape(X)