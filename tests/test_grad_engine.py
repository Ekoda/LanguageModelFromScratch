import pytest
import numpy as np
from src.neural_net.grad_engine import ValueNode


def test_add():
    node1 = ValueNode(3.0)
    node2 = ValueNode(2.0)
    result = node1 + node2
    result.backward()
    assert result.data == 5.0
    assert node1.gradient == 1.0
    assert node2.gradient == 1.0

def test_mul():
    node1 = ValueNode(3.0)
    node2 = ValueNode(2.0)
    result = node1 * node2
    result.backward()
    assert result.data == 6.0
    assert node1.gradient == 2.0
    assert node2.gradient == 3.0

def test_power():
    node = ValueNode(2.0)
    result = node ** 3
    result.backward()
    assert result.data == 8.0
    assert node.gradient == 12.0

def test_log():
    node = ValueNode(2.0)
    result = node.log()
    result.backward()
    assert np.isclose(result.data, np.log(2.0))
    assert node.gradient == 0.5

def test_log_invalid_input():
    node = ValueNode(-1.0)
    with pytest.raises(AssertionError):
        node.log()

def test_relu():
    node1 = ValueNode(-2.0)
    node2 = ValueNode(3.0)
    result1 = node1.relu()
    result2 = node2.relu()
    result1.backward()
    result2.backward()
    assert result1.data == 0.0
    assert result2.data == 3.0
    assert node1.gradient == 0.0
    assert node2.gradient == 1.0

def test_sigmoid():
    node = ValueNode(0.5)
    result = node.sigmoid()
    result.backward()
    assert np.isclose(result.data, 1 / (1 + np.exp(-0.5)))
    assert np.isclose(node.gradient, result.data * (1 - result.data))

def test_sqrt_invalid_input():
    node = ValueNode(-4.0)
    with pytest.raises(AssertionError):
        node.sqrt()

def test_sqrt_zero():
    node = ValueNode(0.0)
    result = node.sqrt()
    result.backward()
    assert result.data == 0.0
    assert node.gradient == 0.0

def test_sqrt():
    node = ValueNode(4.0)
    result = node.sqrt()
    result.backward()
    assert result.data == 2.0
    assert node.gradient == 0.25
