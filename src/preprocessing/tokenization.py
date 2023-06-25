def build_vocab(tokens: list[str]) -> dict[str, int]:
    """
    Builds a vocabulary from a list of tokens.

    Args:
        tokens (list[str]): The list of tokens.

    Returns:
        dict[str, int]: A dictionary mapping tokens to their indices in the vocabulary.
    """
    sorted_unique_tokens = sorted(set(tokens))
    return {word: i for i, word in enumerate(sorted_unique_tokens)}

def tokenize(text: str) -> list[str]:
    """
    Tokenizes a text into a list of tokens.

    Args:
        text (str): The text to tokenize.

    Returns:
        list[str]: A list of tokens.
    """
    return text.split()
