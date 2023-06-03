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