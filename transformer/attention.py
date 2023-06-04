from transformer.neural_network import NeuronLayer

class MultiHeadAttention:
    def __init__ (self, embedding_size: int, n_heads: int = 8):
        self.embedding_size: int = embedding_size
        self.n_heads: int = n_heads


    def forward(self, X: np.ndarray) -> np.ndarray:
        pass


class Head:
    def __init__ (self, embedding_size: int, size: int = 64):
        self.embedding_size: int = embedding_size
        self.size: int = size
        self.query_layer = NeuronLayer(size, activation='linear')
        self.key_layer = None # TODO
        self.value_layer = None # TODO


    def forward(self, X: np.ndarray) -> np.ndarray:
        pass