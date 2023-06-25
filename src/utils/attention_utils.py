import numpy as np
from src.utils.type_utils import Matrix
from src.utils.math_utils import get_shape


def mask_attention_scores(scores: Matrix) -> Matrix:
    """
    Masks the upper triangular part of an attention score matrix, 
    excluding the diagonal, with a very large negative number. 
    This function is applied before softmax in the attention mechanism 
    of Transformer models, ensuring that each word only attends to 
    previous words in the sequence and itself.

    Args:
        scores (Matrix): A 2D list representing raw attention scores. 
            The shape of this matrix should be (T, T) where T is 
            the length of the sequence.

    Returns:
        Matrix: The masked attention scores, ready to be passed into a softmax function.
    """
    n_rows, n_cols = get_shape(scores)
    mask = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(i+1, n_cols):
            mask[i][j] = -1e10
    masked_scores = []
    for score_row, mask_row in zip(scores, mask):
        masked_score_row = []
        for score, mask_val in zip(score_row, mask_row):
            masked_score_row.append(score + mask_val)
        masked_scores.append(masked_score_row)
    return masked_scores
