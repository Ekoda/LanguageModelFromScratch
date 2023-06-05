import numpy as np
from transformer.embedding import generate_embeddings, get_token_embeddings
from transformer.preprocessing.tokenization import tokenize, build_vocab
from transformer.positional_encoding import positional_encoding
from transformer.decoder import Decoder


class EssentialTransformer:
    def __init__(self, data: str, model_dimension: int = 512, n_attention_heads: int = 8, decoder_blocks: int = 6):
        self.data: str = data
        self.vocabulary: dict[str, int] = build_vocab(tokenize(data)) # TODO: better tokenization
        self.embeddings: np.ndarray = generate_embeddings(len(self.vocabulary), model_dimension)
        self.decoder_blocks = [Decoder(model_dimension, n_attention_heads) for _ in range(decoder_blocks)]

    def forward(self, X: str, y: str) -> str:
        input_embeddings = get_token_embeddings(self.embeddings, self.vocabulary, tokenize(X))
        positional_encoded_input = input_embeddings + positional_encoding(len(input_embeddings), input_embeddings.shape[1])
        
        return 0