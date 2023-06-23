import numpy as np
import pytest
from src.utils.math_utils import *
from src.neural_net.grad_engine import ValueNode


def test_sigmoid_activation():
    assert sigmoid_activation(1) == pytest.approx(0.73105857863, rel=1e-9)
    assert sigmoid_activation(-1) == pytest.approx(0.26894142137, rel=1e-9)
    assert sigmoid_activation(0) == 0.5

def test_sigmoid_derivative():
    assert sigmoid_derivative(sigmoid_activation(1)) == pytest.approx(0.19661193324, rel=1e-9)
    assert sigmoid_derivative(sigmoid_activation(-1)) == pytest.approx(0.19661193324, rel=1e-9)
    assert sigmoid_derivative(sigmoid_activation(0)) == 0.25

def test_tanh_activation():
    assert tanh_activation(0) == pytest.approx(0, rel=1e-9)
    assert tanh_activation(1) == pytest.approx(0.7615941559557649, rel=1e-9)
    assert tanh_activation(-1) == pytest.approx(-0.7615941559557649, rel=1e-9)

def test_tanh_derivative():
    assert tanh_derivative(tanh_activation(0)) == pytest.approx(1, rel=1e-9)
    assert tanh_derivative(tanh_activation(1)) == pytest.approx(0.41997434161402603, rel=1e-9)
    assert tanh_derivative(tanh_activation(-1)) == pytest.approx(0.41997434161402603, rel=1e-9)

def test_softmax_without_temp():
    matrix = [[ValueNode(data) for data in row] for row in [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    result = softmax(matrix)
    result_data = [[node.data for node in row] for row in result]
    matrix_data = [[node.data for node in row] for row in matrix]
    expected = np_softmax(matrix_data)
    assert np.allclose(result_data, expected)

def test_softmax_with_temp():
    matrix = [[ValueNode(data) for data in row] for row in [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    result = softmax(matrix, temperature=0.5)
    result_data = [[node.data for node in row] for row in result]
    matrix_data = [[node.data for node in row] for row in matrix]
    expected = np_softmax(matrix_data, temperature=0.5)
    assert np.allclose(result_data, expected)

    result = softmax(matrix, temperature=2)
    result_data = [[node.data for node in row] for row in result]
    expected = np_softmax(matrix_data, temperature=2)
    assert np.allclose(result_data, expected)

def test_dot_product_with_valid_vectors():
    A = [1.0, 2.0, 3.0]
    B = [4.0, 5.0, 6.0]
    expected_result = 32.0
    result = dot(A, B)
    assert result == expected_result

def test_dot_product_with_empty_vectors():
    A, B = [], []
    expected_result = 0.0
    result = dot(A, B)
    assert result == expected_result

def test_dot_product_with_vectors_of_different_lengths():
    A = [1.0, 2.0, 3.0]
    B = [4.0, 5.0]
    with pytest.raises(AssertionError):
        dot(A, B)

def test_get_shape_with_empty_list():
    assert get_shape([]) == ()

def test_get_shape_1d_list():
    assert get_shape([1, 2, 3]) == (3,)

def test_get_shape_2d_list():
    assert get_shape([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == (3, 3)

def test_get_shape_3d_list():
    assert get_shape([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]) == (3, 3, 1)

def test_get_shape_irregular_shapes():
    assert get_shape([[1, 2, 3], [4, 5], [6]]) == (3,)

def test_add_arrays():
    A = [1, 2, 3]
    B = [4, 5, 6]
    expected_result = [5, 7, 9]
    result = add(A, B)
    assert result == expected_result

def test_add_empty_arrays():
    A, B = [], []
    expected_result = []
    result = add(A, B)
    assert result == expected_result

def test_add_arrays_of_different_lengths():
    A = [1, 2, 3]
    B = [4, 5]
    with pytest.raises(AssertionError):
        add(A, B)

def test_add_2d_arrays():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8, 9], [10, 11, 12]]
    expected_result = [[8, 10, 12], [14, 16, 18]]
    result = add(A, B)
    assert result == expected_result

def test_add_2d_arrays_of_different_lengths():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [10, 11]]
    with pytest.raises(AssertionError):
        add(A, B)

def test_matrix_transpose():
    A = [[1, 2, 3], [4, 5, 6]]
    expected_result = [[1, 4], [2, 5], [3, 6]]
    result = transpose(A)
    assert result == expected_result

    A = [[1, 3, 5], [2, 4, 6]]
    expected_result = [[1, 2], [3, 4], [5, 6]]
    result = transpose(A)
    assert result == expected_result

def test_transpose_empty_matrix():
    A: Matrix = []
    expected: Matrix = []
    assert transpose(A) == expected

def test_matmul():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    expected_result = [[58, 64], [139, 154]]
    result = matmul(A, B)

def test_matmul_wrong_dimensions():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10]]
    with pytest.raises(AssertionError):
        matmul(A, B)

def test_apply_elementwise():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    operation = lambda x: x * 2
    expected = [[2, 4, 6], [8, 10, 12], [14, 16, 18]]
    assert apply_elementwise(matrix, operation) == expected

def test_mean_squared_error():
    assert mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert mean_squared_error([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == 1.0
    assert mean_squared_error([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == pytest.approx(4.666, 0.001)

def test_mean_squared_error_different_lengths():
    with pytest.raises(AssertionError):
        mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0])

def test_mean_squared_error_empty_lists():
    with pytest.raises(AssertionError):
        mean_squared_error([], [])