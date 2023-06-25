import math
import numpy as np
import pytest

from src.utils.attention_utils import mask_attention_scores

def isclose_matrix(a, b, rel_tol=1e-9):
    if a is None or b is None or len(a) != len(b):
        return False
    return all(
        all(math.isclose(x, y, rel_tol=rel_tol) for x, y in zip(row_a, row_b))
        for row_a, row_b in zip(a, b)
    )

def compare_numpy_array_and_list(array: np.ndarray, list_: list) -> bool:
    return np.allclose(array, np.array(list_), atol=1e-8)

def test_mask_2x2_matrix():
    scores = [[1, 2], [3, 4]]
    expected_output = [[1, -1e10], [3, 4]]
    assert isclose_matrix(mask_attention_scores(scores), expected_output)

def test_mask_3x3_matrix():
    scores = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected_output = [[1, -1e10, -1e10], [4, 5, -1e10], [7, 8, 9]]
    assert isclose_matrix(mask_attention_scores(scores), expected_output)

def test_mask_single_row_matrix():
    scores = [[1, 2, 3]]
    expected_output = [[1, -1e10, -1e10]]
    assert isclose_matrix(mask_attention_scores(scores), expected_output)

def test_mask_single_column_matrix():
    scores = [[1], [2], [3]]
    expected_output = [[1], [2], [3]]
    assert mask_attention_scores(scores) == expected_output

def test_mask_attention_scores_equivalence_to_numpy():
    scores = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    masked_scores_list = mask_attention_scores(scores)
    masked_scores_numpy = np.array(scores) - np.triu(np.ones((3, 3)), k=1) * 1e10
    assert compare_numpy_array_and_list(masked_scores_numpy, masked_scores_list)

    scores = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    masked_scores_list = mask_attention_scores(scores)
    masked_scores_numpy = np.array(scores) - np.triu(np.ones((4, 4)), k=1) * 1e10
    assert compare_numpy_array_and_list(masked_scores_numpy, masked_scores_list)
