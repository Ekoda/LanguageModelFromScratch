import numpy as np
from src.utils.type_utils import Matrix
from src.neural_net.grad_engine import ValueNode
from src.neural_net.network import NeuralComponent
from src.utils.math_utils import mean, variance, get_shape, add, sub, apply_elementwise


class LayerNorm(NeuralComponent):
    def __init__ (self, size: int):
        self.size = size
        self.gamma = [ValueNode(np.random.randn()) for _ in range(size)]
        self.beta = [ValueNode(np.random.randn()) for _ in range(size)]
        self.epsilon = 1e-6

    def parameters(self):
        return self.gamma + self.beta

    def _compute_means(self, X: Matrix) -> list[ValueNode]:
        return [mean(embedding) for embedding in X]

    def _compute_variances(self, X: Matrix) -> list[ValueNode]:
        return [variance(embedding) for embedding in X]

    def _subtract_means(self, X: Matrix, means: list[ValueNode]) -> Matrix:
        return [[dimension - m for dimension in embedding] for embedding, m in zip(X, means)]

    def _normalize(self, subtracted_mean: Matrix, variances: list[ValueNode]) -> Matrix:
        return [[(dimension / ((v + self.epsilon).sqrt())) for dimension in subtracted_embedding] 
                for subtracted_embedding, v in zip(subtracted_mean, variances)]

    def _scale_and_shift(self, normalized: Matrix) -> Matrix:
        return [[g * dimension + b for g, b, dimension in zip(self.gamma, self.beta, embedding)] 
                for embedding in normalized]

    def forward(self, X: Matrix) -> Matrix:
        means = self._compute_means(X)
        variances = self._compute_variances(X)
        subtracted_mean = self._subtract_means(X, means)
        normalized = self._normalize(subtracted_mean, variances)
        output = self._scale_and_shift(normalized)
        return output
        