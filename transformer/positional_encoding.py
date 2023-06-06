import numpy as np


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
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
    numpy.ndarray
        A seq_len x d_model matrix of positional encodings.
    """
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return PE