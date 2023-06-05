import numpy as np
from typing import Union, Optional
from utils.math_utils import sigmoid_activation, sigmoid_derivative, tanh_activation, tanh_derivative, relu_activation, relu_derivative


class Neuron:
    def __init__(self, n_inputs: int = 1, activation: str = 'tanh', include_bias: bool = True):
        self.w: np.ndarray = np.random.randn(n_inputs) * 0.1
        self.b: float = np.random.randn() * 0.1 if include_bias else 0
        self.include_bias: bool = include_bias
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
        elif self.activation_type == 'relu': 
            return relu_activation(n)
        elif self.activation_type == 'linear':
            return n

    def activation_derivative(self) -> float:
        if self.activation_type == 'sigmoid':
            return sigmoid_derivative(self.output)
        elif self.activation_type == 'tanh':
            return tanh_derivative(self.output)
        elif self.activation_type == 'relu':
            return relu_derivative(self.output)
        elif self.activation_type == 'linear':
            return 1
        
    def compute_gradients(self, upstream_gradient: float) -> None:
        self.gradient = upstream_gradient * self.activation_derivative()
        self.w_gradients = self.gradient * self.inputs

    def update_parameters(self, learning_rate: float) -> None:
        self.w -= learning_rate * self.w_gradients
        if self.include_bias:
            self.b -= learning_rate * self.gradient

    def train (self, upstream_gradient: float, learning_rate: float) -> None:
        self.compute_gradients(upstream_gradient)
        self.update_parameters(learning_rate)

    def forward(self, X: np.ndarray) -> float:
        output = self.activation(np.dot(self.w, X) + self.b)
        self.inputs, self.output = X, output
        return output


class NeuronLayer:
    def __init__(self, size: int, n_inputs: int = 1, activation: str = 'tanh', include_bias: bool = True):
        self.n_neurons = size
        self.neurons = [Neuron(n_inputs, activation, include_bias) for _ in range(size)]

    def train(self, upstream_gradients: list, learning_rate: float):
        for neuron, upstream_gradient in zip(self.neurons, upstream_gradients):
            neuron.train(upstream_gradient, learning_rate)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.array([neuron.forward(X) for neuron in self.neurons])

class  FeedForwardNetwork:
    def __init__ (self, input_size: int, size: int = None):
        self.input_size: int = input_size
        self.size: int = input_size * 4 if size is None else size
        self.layer1 = NeuronLayer(self.size, input_size, activation='relu')
        self.layer2 = NeuronLayer(input_size, self.size, activation='linear')

    def train(self, upstream_gradients: list, learning_rate: float) -> None:
        pass
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        layer1_output = self.layer1.forward(X)
        layer2_output = self.layer2.forward(layer1_output)
        return layer2_output


class LayerNorm:
    def __init__ (self, size: int):
        self.size: int = size
        self.gamma: np.ndarray = np.ones(size)
        self.beta: np.ndarray = np.zeros(size)
        self.gamma_gradients: np.ndarray = np.zeros(size)
        self.beta_gradients: np.ndarray = np.zeros(size)
        self.epsilon: float = 1e-6
        self.normalized_input: Optional[np.ndarray] = None

    def compute_gradients(self, upstream_gradients: list) -> None:
        self.gamma_gradients = np.sum(upstream_gradients * self.normalized_input, axis=0)
        self.beta_gradients = np.sum(upstream_gradients, axis=0)

    def update_parameters(self, learning_rate: float) -> None:
        self.gamma -= learning_rate * self.gamma_gradients
        self.beta -= learning_rate * self.beta_gradients

    def train(self, upstream_gradients: list, learning_rate: float) -> None:
        self.compute_gradients(upstream_gradients)
        self.update_parameters(learning_rate)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=-1, keepdims=True)
        variance = np.var(X, axis=-1, keepdims=True)
        self.normalized_input = (X - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * self.normalized_input + self.beta