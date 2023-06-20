from src.utils.type_utils import Matrix
from src.neural_net.grad_engine import ValueNode
from src.utils.math_utils import dot
import numpy as np


class NeuralComponent:
    def zero_grad(self):
        for p in self.parameters():
            p.gradient = 0

    def parameters(self):
        return []


class Neuron(NeuralComponent):
    def __init__(self, input_size=2, activation='sigmoid', include_bias=True):
        self.w = [ValueNode(np.random.randn()) for _ in range(input_size)]
        self.b = ValueNode(np.random.randn()) if include_bias else None
        self.activation = activation
        self.activation_functions = {
            'sigmoid': self.sigmoid, 
            'relu': self.relu, 
            'linear': self.linear
            }

    def sigmoid(self, x):
        return x.sigmoid()

    def relu(self, x):
        return x.relu()

    def linear(self, x):
        return x

    def parameters(self):
        if self.b:
            return self.w + [self.b]
        return self.w

    def forward(self, X: list[ValueNode]) -> ValueNode:
        pre_activation = dot(X, self.w)
        if self.b:
            pre_activation += self.b
        activation_function = self.activation_functions[self.activation]
        return activation_function(pre_activation)


class NeuronLayer(NeuralComponent):
    def __init__(self, input_size:int, output_size:int, activation:str='relu', include_bias:bool=True):
        self.neurons = [Neuron(input_size, activation, include_bias) for _ in range(output_size)]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def forward(self, X: list[ValueNode]) -> list[ValueNode]:
        return [n.forward(X) for n in self.neurons]


class FeedForwardNetwork(NeuralComponent):
    def __init__(self, input_size:int, output_size:int):
        self.n_inputs = input_size
        self.layers = [NeuronLayer(input_size, output_size * 4, 'relu'), NeuronLayer(output_size * 4, output_size, 'linear')]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def forward(self, X: list[ValueNode]) -> list[ValueNode]:
        for layer in self.layers:
            X = layer.forward(X)
        return X