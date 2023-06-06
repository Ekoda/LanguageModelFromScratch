import pytest
import numpy as np
from transformer.layer_norm import LayerNorm


def test_layer_norm():
    # Test shape
    ln = LayerNorm(size=4)
    X = np.random.randn(4, 4)
    X_norm = ln.forward(X)
    assert X_norm.shape == X.shape
    
    # Test Case 1: Single feature dimension
    ln = LayerNorm(size=1)
    X = np.array([[1, 2, 3, 4, 5]])
    X_norm = ln.forward(X)
    # Expect normalized values close to 0, with some small deviation
    assert np.isclose(np.mean(X_norm), 0)
    assert np.isclose(np.std(X_norm), 1)

    # Test Case 2: Multiple feature dimensions
    ln = LayerNorm(size=4)
    X = np.random.randn(4, 4)
    X_norm = ln.forward(X)
    # Expect normalized values close to 0, with some small deviation
    assert np.allclose(np.mean(X_norm, axis=-1), np.zeros(4))
    assert np.allclose(np.std(X_norm, axis=-1), np.ones(4))

    # Test Case 3: Ensure gamma and beta are applied correctly
    ln = LayerNorm(size=4)
    ln.gamma = np.array([1, 2, 3, 4])
    ln.beta = np.array([1, 2, 3, 4])
    X = np.array([[1, 2, 3, 4]])
    X_norm = ln.forward(X)
    # Expect result to be gamma * normalized X + beta
    assert np.allclose(X_norm, ln.gamma * (X - np.mean(X)) / np.std(X) + ln.beta)

def test_layer_norm_backprop():
        np.random.seed(0)  # Ensure reproducibility
        layer_norm = LayerNorm(5)  # Size 5 for simplicity
        X = np.random.randn(10, 5)  # Batch of 10 examples

        Y = layer_norm.forward(X)
        upstream_gradients = np.random.randn(*Y.shape)
        layer_norm.compute_gradients(upstream_gradients)

        # Check gamma gradients
        for i in range(layer_norm.size):
            # Perturb gamma[i]
            old_gamma_i = layer_norm.gamma[i]
            layer_norm.gamma[i] += 1e-5

            # New forward pass
            Y_new = layer_norm.forward(X)

            # Revert gamma[i]
            layer_norm.gamma[i] = old_gamma_i

            # Expected change in output
            expected_change = np.sum(upstream_gradients * (Y_new - Y))

            # Actual change (according to gradient)
            actual_change = layer_norm.gamma_gradients[i] * 1e-5

            # They should be approximately equal
            assert np.isclose(expected_change, actual_change, atol=1e-3)

        # Check beta gradients (similar process)
        for i in range(layer_norm.size):
            old_beta_i = layer_norm.beta[i]
            layer_norm.beta[i] += 1e-5
            Y_new = layer_norm.forward(X)
            layer_norm.beta[i] = old_beta_i
            expected_change = np.sum(upstream_gradients * (Y_new - Y))
            actual_change = layer_norm.beta_gradients[i] * 1e-5
            assert np.isclose(expected_change, actual_change, atol=1e-3)