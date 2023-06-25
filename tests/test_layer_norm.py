import pytest
from src.neural_net.grad_engine import ValueNode
from src.transformer.components.layer_norm import LayerNorm
from src.utils.data_utils import mock_matrix
from src.utils.math_utils import get_shape


def test_init():
    layer_norm = LayerNorm(64)
    assert layer_norm.size == 64
    assert get_shape(layer_norm.gamma) == (64,)
    assert get_shape(layer_norm.beta) == (64,)
    assert layer_norm.epsilon == 1e-6

def test_compute_means_and_variances():
    layer_norm = LayerNorm(64)
    matrix = mock_matrix(8, 64)
    means = layer_norm._compute_means(matrix)
    variances = layer_norm._compute_variances(matrix)
    assert get_shape(means) == (8,)
    assert get_shape(variances) == (8,)
    assert all([isinstance(mean, ValueNode) for mean in means])
    assert all([isinstance(var, ValueNode) for var in variances])

def test_subtract_means():
    layer_norm = LayerNorm(64)
    matrix = mock_matrix(8, 64)
    means = layer_norm._compute_means(matrix)
    subtracted_mean = layer_norm._subtract_means(matrix, means)
    assert get_shape(subtracted_mean) == (8, 64)

def test_normalize():
    layer_norm = LayerNorm(64)
    matrix = mock_matrix(8, 64)
    means = layer_norm._compute_means(matrix)
    variances = layer_norm._compute_variances(matrix)
    subtracted_mean = layer_norm._subtract_means(matrix, means)
    normalized = layer_norm._normalize(subtracted_mean, variances)
    assert get_shape(normalized) == (8, 64)

def test_scale_and_shift():
    layer_norm = LayerNorm(64)
    matrix = mock_matrix(8, 64)
    means = layer_norm._compute_means(matrix)
    variances = layer_norm._compute_variances(matrix)
    subtracted_mean = layer_norm._subtract_means(matrix, means)
    normalized = layer_norm._normalize(subtracted_mean, variances)
    scaled_shifted = layer_norm._scale_and_shift(normalized)
    assert get_shape(scaled_shifted) == (8, 64)

def test_forward():
    layer_norm = LayerNorm(64)
    matrix = mock_matrix(8, 64)
    output = layer_norm.forward(matrix)
    assert get_shape(output) == (8, 64)
