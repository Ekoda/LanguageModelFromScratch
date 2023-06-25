import numpy as np
import pytest

from src.transformer.components.positional_encoding import sinusoidal_encodings
from src.utils.math_utils import get_shape


def test_positional_encoding():
    seq_len = 10
    d_model = 16
    pos_enc = sinusoidal_encodings(seq_len, d_model)
    assert get_shape(pos_enc) == (seq_len, d_model)

def test_positional_encoding_values():
    PE = sinusoidal_encodings(10, 16)
    assert all(-1 <= value <= 1 for row in PE for value in row)
