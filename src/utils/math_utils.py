import numpy as np


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

def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """
    Compute the softmax of each element along an axis of a numpy array.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, optional
        Axis along which the softmax normalization is applied. The default is -1.

    Returns
    -------
    np.ndarray
        The array with softmax applied elementwise along the specified axis.

    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def softmax_with_temperature(x: np.ndarray, temperature: float) -> np.ndarray:
    """
    Compute the softmax of each element along an axis of a numpy array 
    with temperature scaling applied.

    Parameters
    ----------
    logits : np.ndarray
        Input array (logits).
    temperature : float
        The temperature factor to scale the logits. Higher values make the output 
        probabilities closer to uniform distribution (more randomness),
        and lower values make it closer to one-hot encoding (less randomness).

    Returns
    -------
    np.ndarray
        The array with temperature-scaled softmax applied.
    """
    scaled_logits = x / temperature
    e_x = np.exp(scaled_logits - np.max(scaled_logits))
    return e_x / e_x.sum()


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