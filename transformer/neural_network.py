import numpy as np
from typing import Union, Optional
from utils.math_utils import sigmoid_activation, sigmoid_derivative, tanh_activation, tanh_derivative


class Neuron:
    def __init__(self, n_inputs: int = 1, activation: str = 'tanh'):
        self.w: np.ndarray = np.random.randn(n_inputs) * 0.1
        self.b: float = np.random.randn() * 0.1
        self.activation_type: str = activation
        self.gradient: float = 0
        self.w_gradients: np.ndarray = np.zeros(n_inputs)
        self.output: Optional[float] = None
        self.inputs: Optional[np.ndarray] = None

    def activation(self, n: float) -> float:
        if self.activation_type == 'sigmoid':
            return sigmoid_activation(n)
        elif self.activation_type == 'tanh':
            return np.tanh(n)
        elif self.activation_type == 'linear':
            return n

    def activation_derivative(self) -> float:
        if self.activation_type == 'sigmoid':
            return sigmoid_derivative(self.output)
        elif self.activation_type == 'tanh':
            return tanh_derivative(self.output)
        elif self.activation_type == 'linear':
            return 1
        
    def compute_gradients(self, upstream_gradient: float) -> None:
        self.gradient = upstream_gradient * self.activation_derivative()
        self.w_gradients = self.gradient * self.inputs

    def update_parameters(self, learning_rate: float) -> None:
        self.w -= learning_rate * self.w_gradients
        self.b -= learning_rate * self.gradient

    def forward(self, X: np.ndarray) -> float:
        output = self.activation(np.dot(self.w, X) + self.b)
        self.inputs, self.output = X, output
        return output

class NeuronLayer:
    def __init__(self, size: int, n_inputs: int = 1, activation: str = 'tanh'):
        self.n_neurons = size
        self.neurons = [Neuron(n_inputs, activation) for _ in range(size)]

    def train(self, upstream_gradients: list, learning_rate: float):
        for neuron, upstream_gradient in zip(self.neurons, upstream_gradients):
            neuron.compute_gradients(upstream_gradient)
            neuron.update_parameters(learning_rate)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.array([neuron.forward(X) for neuron in self.neurons])