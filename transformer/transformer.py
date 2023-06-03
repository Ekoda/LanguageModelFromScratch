import numpy as np
from transformer.embedding import generate_embeddings, get_token_embeddings
from transformer.preprocessing.tokenization import tokenize, build_vocab
from transformer.positional_encoding import positional_encoding


class EssentialTransformer:
    def __init__(self, data: str, embedding_size: int = 512):
        self.data: str = data
        self.vocabulary: dict[str, int] = build_vocab(tokenize(data)) # TODO: better tokenization
        self.embeddings: np.ndarray = generate_embeddings(len(self.vocabulary), embedding_size)

    def forward(self, X: str, y: str) -> str:
        input_embeddings = get_token_embeddings(self.embeddings, self.vocabulary, tokenize(X))
        positional_encoded_input = input_embeddings + positional_encoding(len(input_embeddings), input_embeddings.shape[1])

        # TODO Attention block
        # TODO Add & Norm block
        # TODO Feed Forward block
        # TODO Linear layer

        return 0