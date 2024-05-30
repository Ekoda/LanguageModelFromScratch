# Language Model from Scratch
A Language Model built from scratch, in order to explore and understand the Transformer model, specifically a fully functional decoder for next-word prediction, similar to GPT. This model is implemented with backpropagation, enabling actual training, albeit at a very slow pace. The focus is on clarity and education rather than optimization

---
## Example
A concrete illustration of the transformer can be found in the `example.ipynb` notebook. Here the model is trained on the sentence "The quick fox jumped over the lazy dog.", which it manages to predict perfectly. A fairly useless application, I admit, but a proof of concept nonetheless. Scaling the model up to 175 billion parameters with internet scale datasets is left as an exercise to the reader.

---
## How to Use
The core component is the `EssentialTransformer` class, which handles all the intricacies of training and making predictions.

```python
from src.transformer.model import EssentialTransformer
from src.utils.generic_utils import load_text_data

# Load data
data = load_text_data("data/test.txt")

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
