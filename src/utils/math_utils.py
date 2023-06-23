import numpy as np
from src.utils.type_utils import Matrix, ValueNode
from math import exp


def sigmoid_activation(n: float) -> float:
    return 1 / (1 + np.exp(-n))


def sigmoid_derivative(sigmoid_output: float) -> float:
        """
        Calculate the derivative of the sigmoid function. d/d_sigmoid_output.

        This function assumes that 'sigmoid_output' is the output of a sigmoid function.
        The derivative of the sigmoid function is sigmoid_output * (1 - sigmoid_output).

        Args:
            sigmoid_output (float): The output of the sigmoid function.

        Returns:
            float: The derivative of the sigmoid function at the corresponding input.
        """
        return  sigmoid_output * (1 - sigmoid_output)

def tanh_activation(n: float) -> float:
    return np.tanh(n)

def tanh_derivative(tanh_output: float) -> float:
    """
    Calculate the derivative of the tanh function. d/d_tanh_output.

    This function assumes that 'tanh_output' is the output of a tanh function.
    The derivative of the tanh function is 1 - tanh_output^2.

    Args:
        tanh_output (float): The output of the tanh function.

    Returns:
        float: The derivative of the tanh function at the corresponding input.
    """
    return 1 - tanh_output ** 2

def matrix_max(matrix: Matrix) -> ValueNode:
    """Finds the maximum value in a 2D list (matrix)"""
    return max(max(row) for row in matrix)

def matrix_sum(matrix: Matrix) -> ValueNode:
    """Finds the sum of all values in a 2D list (matrix)"""
    return sum(sum(row) for row in matrix)

def np_softmax(matrix: np.ndarray, temperature: float = None) -> np.ndarray:
        if temperature is not None:
            matrix = np.array(matrix) / temperature
        e_x = np.exp(matrix - np.max(matrix))
        return e_x / e_x.sum()

def softmax(matrix: Matrix, temperature: float = None) -> Matrix:
    """
    Compute the softmax of each element of a 2D list (matrix)

    If temperature is provided, computes the softmax of each element with temperature scaling applied.

    Parameters
    ----------
    matrix : Matrix
        Input matrix.
    temperature : float, optional
        The temperature factor to scale the logits. Higher values make the output 
        probabilities closer to uniform distribution (more randomness),
        and lower values make it closer to one-hot encoding (less randomness).

    Returns
    -------
    Matrix
        The matrix with softmax applied elementwise.
    """
    if temperature:
        scaled_logits = apply_elementwise(matrix, lambda x: x / temperature)
        max_value = matrix_max(scaled_logits)
        e_x = apply_elementwise(scaled_logits, lambda x: (x - max_value).exp())
    else:
        max_value = matrix_max(matrix)
        e_x = apply_elementwise(matrix, lambda x: (x - max_value).exp())
    sum_e_x = matrix_sum(e_x)
    return apply_elementwise(e_x, lambda x: x / sum_e_x)

def relu_activation(n: float) -> float:
    """
    ReLU activation function.

    Args:
        n (float): Input value.

    Returns:
        float: Activation of the input.
    """
    return np.maximum(0, n)


def relu_derivative(relu_output: float) -> float:
    """
    Calculate the derivative of the ReLU function.

    This function assumes that 'relu_output' is the output of a ReLU function.
    The derivative of the ReLU function is 1 for relu_output > 0, and 0 otherwise.

    Args:
        relu_output (float): The output of the ReLU function.

    Returns:
        float: The derivative of the ReLU function at the corresponding input.
    """
    return (relu_output > 0) * 1


def dot(A: list[float], B: list[float]) -> float:
    assert len(A) == len(B), "Dot product requires arrays of the same length"
    return sum(a * b for a, b in zip(A, B))

def mean(X: list[float]) -> float:
    return sum(X) / len(X)

def variance(X: list[float]) -> float:
    return sum((x - mean(X)) ** 2 for x in X) / len(X)

def mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    assert len(y_true) != 0, "y_true and y_pred must not be empty"
    return sum([(y - y_hat ) ** 2 for y, y_hat in zip(y_true, y_pred)]) / len(y_true)

def get_shape(li) -> tuple[int, ...]:
    if not isinstance(li, list) or not li:
        return ()
    if all(isinstance(item, list) and len(item) == len(li[0]) for item in li):
        return (len(li),) + get_shape(li[0])
    else:
        return (len(li),)

def add(A: Matrix, B: Matrix) -> Matrix:
    """
    Add two lists (1D or 2D) element-wise. 

    This function performs element-wise addition of two lists, A and B. Both lists should 
    have the same size (length for 1D lists, or same number of rows and columns for 2D lists). 
    The function supports 1D lists (vectors) and 2D lists (matrices), and it determines which 
    operation to perform based on the dimensionality of the inputs.

    Parameters
    ----------
    A : list
        First list for addition. Can be 1D or 2D.
    B : list
        Second list for addition. Should have the same dimensions as A.

    Returns
    -------
    list
        The resulting list after performing element-wise addition. Same dimensions as the inputs.

    Raises
    ------
    AssertionError
        If A and B have different sizes, an AssertionError is raised.
    """
    assert len(A) == len(B), "Addition requires arrays of the same size" 
    if all(isinstance(x, list) for x in A) and all(isinstance(x, list) for x in B):
        assert len(A) == len(B), "Addition requires arrays of the same size"
        assert all(len(a) == len(b) for a, b in zip(A, B)), "All rows must be the same length"
        return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]
    else:
        assert len(A) == len(B), "Addition requires arrays of the same length"
        return [a + b for a, b in zip(A, B)]

def sub(A: Matrix, B: Matrix) -> Matrix:
    """
    Add two lists (1D or 2D) element-wise. 

    This function performs element-wise subtraction of two lists, A and B. Both lists should 
    have the same size (length for 1D lists, or same number of rows and columns for 2D lists). 
    The function supports 1D lists (vectors) and 2D lists (matrices), and it determines which 
    operation to perform based on the dimensionality of the inputs.

    Parameters
    ----------
    A : list
        First list for subtraction. Can be 1D or 2D.
    B : list
        Second list for subtraction. Should have the same dimensions as A.

    Returns
    -------
    list
        The resulting list after performing element-wise subtraction. Same dimensions as the inputs.

    Raises
    ------
    AssertionError
        If A and B have different sizes, an AssertionError is raised.
    """
    assert len(A) == len(B), "subtraction requires arrays of the same size" 
    if all(isinstance(x, list) for x in A) and all(isinstance(x, list) for x in B):
        assert len(A) == len(B), "subtraction requires arrays of the same size"
        assert all(len(a) == len(b) for a, b in zip(A, B)), "All rows must be the same length"
        return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]
    else:
        assert len(A) == len(B), "subtraction requires arrays of the same length"
        return [a - b for a, b in zip(A, B)]

def transpose(A: Matrix) -> Matrix:
    if not A or not A[0]:
        return []
    m, n = len(A), len(A[0])
    transposed_matrix = []
    for i in range(n):
        transposed_row = []
        for j in range(m):
            transposed_row.append(A[j][i])
        transposed_matrix.append(transposed_row)
    return transposed_matrix

def matmul(A: Matrix, B: Matrix) -> Matrix:
    Am, An = len(A), len(A[0])
    Bm, Bn = len(B), len(B[0])
    assert An == Bm, "Matrix multiplication requires the number of columns in A to be equal to the number of rows in B"
    result_matrix = []
    for i in range(Am):
        result_row = []
        for j in range(Bn):
            vector_b = [B[k][j] for k in range(Bm)]
            dot_product = dot(A[i], vector_b)
            result_row.append(dot_product)
        result_matrix.append(result_row)
    return result_matrix

def apply_elementwise(matrix: Matrix, operation: callable) -> Matrix:
    return [[operation(x) for x in row] for row in matrix]
