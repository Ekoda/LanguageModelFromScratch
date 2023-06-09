import numpy as np
from src.transformer.components.embedding import generate_embeddings, get_token_embeddings
from src.transformer.preprocessing.tokenization import tokenize, build_vocab
from src.transformer.components.positional_encoding import positional_encoding
from src.transformer.decoder import Decoder
from src.transformer.components.neural_network import NeuronLayer
from src.utils.math_utils import softmax_with_temperature
from src.utils.data_utils import find_next_word


class EssentialTransformer:
    def __init__(self, data: str, model_dimension: int = 512, n_attention_heads: int = 8, decoder_blocks: int = 6):
        self.data: str = data
        self.vocabulary: dict[str, int] = build_vocab(tokenize(data)) # TODO: better tokenization
        self.reversed_vocabulary: dict[int, str] = {index: word for word, index in self.vocabulary.items()}
        self.embeddings: np.ndarray = generate_embeddings(len(self.vocabulary), model_dimension)
        self.decoder_blocks = [Decoder(model_dimension, n_attention_heads) for _ in range(decoder_blocks)]
        self.final_layer = NeuronLayer(len(self.vocabulary), model_dimension, activation='linear')

    def calculate_loss(self, prediction, target):
        pass

    def train(self, X, y):
        pass

    def forward(self, X: str, y: str, temperature: float = 1e-9) -> str:
        input_embeddings = get_token_embeddings(self.embeddings, self.vocabulary, tokenize(X))
        positional_encoded_input = input_embeddings + positional_encoding(len(input_embeddings), input_embeddings.shape[1])
        decoder_output = positional_encoded_input
        for decoder in self.decoder_blocks:
            decoder_output = decoder.forward(decoder_output)
        final_layer_output = np.array([self.final_layer.forward(embedding) for embedding in decoder_output])
        word_probabilities = softmax_with_temperature(final_layer_output, temperature)
        return find_next_word(word_probabilities, self.reversed_vocabulary)