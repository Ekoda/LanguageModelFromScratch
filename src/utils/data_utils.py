import numpy as np
from src.utils.type_utils import Matrix
from src.neural_net.grad_engine import ValueNode
from src.utils.type_utils import Matrix

def find_next_word(probabilities: Matrix, index_to_word: dict[int, str]) -> str:
    next_word_index = np.argmax(probabilities[-1])
    return index_to_word[next_word_index]

def mock_matrix(rows: int, cols: int) -> Matrix:
    return [[ValueNode(np.random.rand()) for _ in range(cols)] for _ in range(rows)]

def sequence_data(tokens: list[str], sequence_length: int) -> list[list[str]]:
    sequences = []
    for i in range(len(tokens)):
        if i % sequence_length == 0:
            sequences.append(tokens[i:i + sequence_length])
    return sequences