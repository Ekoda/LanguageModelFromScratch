from src.utils.math_utils import get_shape
from src.utils.type_utils import Matrix


def load_text_data(path: str) -> str:
    with open(path, 'r') as file:
        data = file.read()
    return data

def print_matrix(matrix: Matrix, num_decimals: int = 2) -> None:
    if not matrix:
        print("Empty matrix")
        return
    formatted_matrix = [[f"{value.data:.{num_decimals}f}" if value.data > -1e9 else f"{value.data:.0e}" for value in row] for row in matrix]
    max_len = max(len(str_) for row in formatted_matrix for str_ in row)
    n_rows, n_cols = get_shape(matrix)
    value_type = type(matrix[0][0]).__name__ if n_rows > 0 and n_cols > 0 else "None"
    info_string = f" Shape: {n_rows}x{n_cols}, Type: {value_type} "
    width = max((len(formatted_matrix[0]) * (max_len + 3)), len(info_string) + 4)
    print("┌" + "─" * width + "┐")
    print("|" + info_string.center(width) + "|")
    print("|" + " " * width + "|")
    for row in formatted_matrix:
        print("|", end="")
        for formatted_value in row:
            print(f" {formatted_value: <{max_len}} |", end="")
        print()
    print("|" + " " * width + "|")
    print("└" + "─" * width + "┘")
