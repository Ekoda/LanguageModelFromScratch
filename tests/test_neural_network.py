import pytest
import numpy as np
from src.neural_net.network import FeedForwardNetwork, Neuron, NeuronLayer
from src.neural_net.grad_engine import ValueNode


def test_Neuron():
    neuron = Neuron(input_size=3, activation='sigmoid')
    
    assert len(neuron.parameters()) == 4  # 3 weights and 1 bias.

    X = [ValueNode(np.random.randn()) for _ in range(3)]
    output = neuron.forward(X)
    assert isinstance(output, ValueNode)


def test_NeuronLayer():
    layer = NeuronLayer(input_size=3, output_size=2, activation='relu')
    
    assert len(layer.parameters()) == 8  # 2 neurons * (3 weights and 1 bias).

    X = [ValueNode(np.random.randn()) for _ in range(3)]
    output = layer.forward(X)
    assert isinstance(output, list)
    assert len(output) == 2
    assert all(isinstance(o, ValueNode) for o in output)

    layer = NeuronLayer(input_size=3, output_size=2, activation='relu', include_bias=False)
    output = layer.forward(X)
    assert len(layer.parameters()) == 6  # 2 neurons * 3 weights.
    assert len(output) == 2


def test_FeedForwardNetwork():
    network = FeedForwardNetwork(input_size=3, output_size=2)
    
    assert len(network.parameters()) == 50 # (3 * 2 * 4) + (2 * 4) + (2 * 4 * 2) + 2

    X = [ValueNode(np.random.randn()) for _ in range(3)]
    output = network.forward(X)
    assert isinstance(output, list)
    assert len(output) == 2
    assert all(isinstance(o, ValueNode) for o in output)