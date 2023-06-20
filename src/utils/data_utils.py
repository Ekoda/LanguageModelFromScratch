import numpy as np
from src.utils.type_utils import Matrix
from src.neural_net.grad_engine import ValueNode

def find_next_word(probabilities: np.ndarray, index_to_word: dict[int, str]) -> str:
    next_word_index = np.argmax(probabilities[-1])
    return index_to_word[next_word_index]

def mock_matrix(rows: int, cols: int) -> Matrix:
    return [[ValueNode(np.random.rand()) for _ in range(cols)] for _ in range(rows)]