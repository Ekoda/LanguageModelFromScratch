import numpy as np

def mask_attention_scores(scores: np.ndarray) -> np.ndarray:
    """
    Masks the upper triangular part of an attention score matrix, 
    excluding the diagonal, with a very large negative number. 
    This function is applied before softmax in the attention mechanism 
    of Transformer models, ensuring that each word only attends to 
    previous words in the sequence and itself.

    Parameters
    ----------
    scores: np.ndarray
        A 2D numpy array representing raw attention scores. 
        The shape of this array should be (T, T) where T is 
        the length of the sequence.

    Returns
    -------
    np.ndarray
        The masked attention scores, ready to be passed into a softmax function.
    """
    rows, cols = scores.shape
    mask = np.triu(np.ones((rows, cols)), k=1)
    return scores - (mask * 1e10)