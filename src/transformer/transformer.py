import numpy as np
from src.transformer.components.embedding import generate_embeddings, get_token_embeddings
from src.transformer.preprocessing.tokenization import tokenize, build_vocab
from src.transformer.components.positional_encoding import encode_position
from src.transformer.decoder import Decoder
from src.neural_net.network import NeuronLayer
from src.utils.math_utils import softmax, get_shape
from src.utils.data_utils import find_next_word


class EssentialTransformer:
    def __init__(self, data: str, model_dimension: int = 512, n_attention_heads: int = 8, decoder_blocks: int = 6):
        self.data: str = data
        self.model_dimension: int = model_dimension
        self.vocabulary: dict[str, int] = build_vocab(tokenize(data)) # TODO: better tokenization
        self.reversed_vocabulary: dict[int, str] = {index: word for word, index in self.vocabulary.items()}
        self.embeddings: list[list[float]] = generate_embeddings(len(self.vocabulary), model_dimension)
        self.decoder_blocks = [Decoder(model_dimension, n_attention_heads) for _ in range(decoder_blocks)]
        self.output_layer = NeuronLayer(model_dimension, len(self.vocabulary), activation='linear')

    def calculate_loss(self, prediction, target):
        pass

    def train(self, X, y):
        pass

    def forward(self, X: str, y: str = None, temperature: float = None) -> str:
        sequence_embeddings = get_token_embeddings(self.embeddings, self.vocabulary, tokenize(X))
        positionally_encoded_embeddings = encode_position(sequence_embeddings)
        decoder_output = positionally_encoded_embeddings
        for decoder in self.decoder_blocks:
            decoder_output = decoder.forward(decoder_output)
        output_layer = [self.output_layer.forward(embedding) for embedding in decoder_output]
        token_predictions = softmax(output_layer, temperature=temperature)
        return find_next_word(token_predictions, self.reversed_vocabulary)