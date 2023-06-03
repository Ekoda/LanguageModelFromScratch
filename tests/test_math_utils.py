import numpy as np
import pytest
from utils.math_utils import sigmoid_activation, sigmoid_derivative, tanh_activation, tanh_derivative

def test_sigmoid_activation():
    assert sigmoid_activation(1) == pytest.approx(0.73105857863, rel=1e-9)
    assert sigmoid_activation(-1) == pytest.approx(0.26894142137, rel=1e-9)
    assert sigmoid_activation(0) == 0.5

def test_sigmoid_derivative():
    assert sigmoid_derivative(sigmoid_activation(1)) == pytest.approx(0.19661193324, rel=1e-9)
    assert sigmoid_derivative(sigmoid_activation(-1)) == pytest.approx(0.19661193324, rel=1e-9)
    assert sigmoid_derivative(sigmoid_activation(0)) == 0.25

def test_tanh_activation():
    assert tanh_activation(0) == pytest.approx(0, rel=1e-9)
    assert tanh_activation(1) == pytest.approx(0.7615941559557649, rel=1e-9)
    assert tanh_activation(-1) == pytest.approx(-0.7615941559557649, rel=1e-9)

def test_tanh_derivative():
    assert tanh_derivative(tanh_activation(0)) == pytest.approx(1, rel=1e-9)
    assert tanh_derivative(tanh_activation(1)) == pytest.approx(0.41997434161402603, rel=1e-9)
    assert tanh_derivative(tanh_activation(-1)) == pytest.approx(0.41997434161402603, rel=1e-9)