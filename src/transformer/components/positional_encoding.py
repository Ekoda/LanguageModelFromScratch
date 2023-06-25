import numpy as np

from src.utils.math_utils import add
from src.utils.type_utils import Matrix


def sinusoidal_encodings(seq_len: int, d_model: int) -> Matrix:
    """
    Compute sinusoidal positional encoding.
    Parameters
    ----------
    seq_len: int
        The length of sequences.
    d_model: int
        The dimension of the model.
    Returns
    -------
    list[list[float]]

        A seq_len x d_model matrix of positional encodings.
    """
    PE = [[0 for _ in range(d_model)] for _ in range(seq_len)]
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos][i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                PE[pos][i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return PE


def encode_position(sequence_embeddings: Matrix) -> Matrix:
    """
    Encode the position of each token in a sequence.
    Parameters
    ----------
    sequence_embeddings: list[list[float]]
        A seq_len x d_model matrix of token embeddings.
    Returns
    -------
    list[list[float]]
        A seq_len x d_model matrix of positionally encoded embeddings.
    """
    seq_len, d_model = len(sequence_embeddings), len(sequence_embeddings[0])
    encodings = sinusoidal_encodings(seq_len, d_model)
    return add(sequence_embeddings, encodings)
