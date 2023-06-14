from src.neural_net.grad_engine import ValueNode
from src.utils.math_utils import dot


class NeuralComponent:
    def zero_grad(self):
        for p in self.parameters():
            p.gradient = 0

    def parameters(self):
        return []


class Neuron(NeuralComponent):
    def __init__(self, input_size=2, activation='sigmoid'):
        self.w = [ValueNode(np.random.randn() * 0.01) for _ in range(input_size)]
        self.b = ValueNode(np.random.randn() * 0.01)
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
        return self.w + [self.b]

    def forward(self, X):
        pre_activation = dot(X, self.w) + self.b
        activation_function = self.activation_functions[self.activation]
        return activation_function(pre_activation)


class NeuronLayer(NeuralComponent):
    def __init__(self, input_size:int, output_size:int, activation:str, loss:str='binary_cross_entropy'):
        self.neurons = [Neuron(input_size, activation) for _ in range(output_size)]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def forward(self, X: list[ValueNode]) -> list[ValueNode]:
        return [n.forward(X) for n in self.neurons]


class FeedForwardNetwork(NeuralComponent):
    def __init__(self, input_size:int, output_size:int):
        self.n_inputs = input_size
        self.n_outputs = output_size
        self.size: int = input_size * 4
        self.layers = [NeuronLayer(input_size, self.size, 'relu'), NeuronLayer(self.size, output_size, 'linear')]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def forward(self, X: list[ValueNode]) -> list[ValueNode]:
        for layer in self.layers:
            X = layer.forward(X)
        return X