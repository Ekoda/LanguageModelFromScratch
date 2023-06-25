# EssentialTransformer
The Transformer model has revolutionized the field of natural language processing, enabling state-of-the-art results in tasks like translation, summarization, and much more.

EssentialTransformer is an exploration into the intricate workings of the Transformer model. A fully functional decoder only next word prediction model in the style of GPT, built entirely from scratch. This project is all about building intuition and understanding of the fundamental concepts that underpin the Transformer architecture. The focus is not to be the fastest or most optimized model. Instead its aim is to be clear and educational for those wanting to understand the technical details of how transformers work.

To achieve this end the Transformer is broken down into its core components, each piece implemented separately.

---
## How to Use
The core component is the `EssentialTransformer` class, which handles all the intricacies of training and making predictions.

```python
from src.utils.generic_utils import load_text_data
from src.transformer.model import EssentialTransformer

# Load data
data = load_text_data(path="data/test.txt")

# Initialize and train the model
model = EssentialTransformer(data, model_dimension=512, n_attention_heads=8, decoder_blocks=6)
model.train(data, sequence_length=8, epochs=2)

# Make predictions
prediction = model.forward("the quick brown fox jumps")
```

---
## Project Structure
The project is structured in a hierarchical and modular fashion according to the original "attention is all you need" paper (Vaswani et al., 2017). As such the code in the components folder contain most of the detail, while code such as the model.py contain the transformer which ties all the pieces together. The neural_net folder contains a mini neural network library, complete with a grad engine which is really an extension of Andrej Karpathy's micrograd, for details I refer back to his brilliant material and repo at: https://github.com/karpathy/micrograd

---
## Requirements
- Python 3.10 or later

Dependencies are listed in the `requirements.txt` file. To install these dependencies, navigate to the project directory and run the following command:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

The aim with the project was to be as barebones as possible, using numpy only for mathematic operations. Even going as far as implementing tensor operations when needed. At one point building a tensor / linear algebra library for the grad engine was consider but was opted out of in order to stay on track.