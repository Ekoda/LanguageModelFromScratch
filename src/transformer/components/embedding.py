import numpy as np


def generate_embeddings(vocabulary_size: int, embedding_size: int) -> np.ndarray:
    """
    Generates a matrix of initial embeddings for a vocabulary of given size and embedding dimension.

    Args:
        vocabulary_size (int): The number of words in the vocabulary.
        embedding_size (int): The dimension of the embedding space.

    Returns:
        np.ndarray: A matrix of shape (vocabulary_size, embedding_size) containing the initial embeddings.
    """
    return np.random.rand(vocabulary_size, embedding_size) * 0.01

def get_token_embeddings(embeddings: np.ndarray, vocabulary: dict[str, int], tokens: list[str]) -> np.ndarray:
    """
    Gets the embeddings for a list of tokens from a matrix of embeddings.

    Args:
        embeddings (np.ndarray): A matrix of shape (vocabulary_size, embedding_size) containing the embeddings.
        vocabulary (dict[str, int]): A dictionary mapping tokens to their indices in the vocabulary.
        tokens (list[str]): The list of tokens to fetch the embeddings for.

    Returns:
        np.ndarray: A matrix of shape (len(tokens), embedding_size) containing the embeddings for the tokens.
    """
    return np.array([embeddings[vocabulary[token]] for token in tokens])