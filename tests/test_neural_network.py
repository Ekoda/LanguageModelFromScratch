import pytest
import numpy as np
from transformer.neural_network import NeuronLayer, Neuron

def test_neuron():
    neuron = Neuron(n_inputs=2, activation='sigmoid')

    # Forward test
    output = neuron.forward(np.array([1, 1]))
    assert 0 <= output <= 1, "Output of sigmoid activation should be between 0 and 1"

    # Gradient computation and parameter update test
    upstream_gradient = 0.1
    neuron.compute_gradients(upstream_gradient)
    old_w = neuron.w.copy()
    old_b = neuron.b
    learning_rate = 0.01

    neuron.update_parameters(learning_rate)
    assert all(neuron.w < old_w), "Weights should be updated (decreased due to gradient descent)"
    assert neuron.b < old_b, "Bias should be updated (decreased due to gradient descent)"

def test_neuron_layer():
    neuron_layer = NeuronLayer(size=2, n_inputs=2, activation='sigmoid')

    # Forward test
    outputs = neuron_layer.forward(np.array([1, 1]))
    assert len(outputs) == 2, "Should have 2 outputs for 2 neurons"
    assert all(0 <= output <= 1 for output in outputs), "All outputs of sigmoid activation should be between 0 and 1"

    # Training (gradient computation and parameter update) test
    upstream_gradients = [0.1, 0.2]
    old_weights = [neuron.w.copy() for neuron in neuron_layer.neurons]
    old_biases = [neuron.b for neuron in neuron_layer.neurons]
    learning_rate = 0.01

    neuron_layer.train(upstream_gradients, learning_rate)
    for i, neuron in enumerate(neuron_layer.neurons):
        assert all(neuron.w < old_weights[i]), f"Weights of neuron {i} should be updated (decreased due to gradient descent)"
        assert neuron.b < old_biases[i], f"Bias of neuron {i} should be updated (decreased due to gradient descent)"
