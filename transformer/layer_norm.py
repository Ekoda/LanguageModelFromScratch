import numpy as np
from typing import Optional


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
        