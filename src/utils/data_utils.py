import numpy as np

def find_next_word(probabilities: np.ndarray, index_to_word: dict[int, str]) -> str:
    next_word_index = np.argmax(probabilities[-1])
    return index_to_word[next_word_index]