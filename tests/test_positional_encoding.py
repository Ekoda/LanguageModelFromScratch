import numpy as np
import pytest
from src.transformer.components.positional_encoding import positional_encoding

def test_positional_encoding():
    seq_len = 10
    d_model = 16
    pos_enc = positional_encoding(seq_len, d_model)
    assert pos_enc.shape == (seq_len, d_model)