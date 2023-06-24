# EssentialTransformer
The Transformer model has revolutionized the field of natural language processing, enabling state-of-the-art results in tasks like translation, summarization, and much more.

EssentialTransformer is an exploration into the intricate workings of the Transformer model. A decoder only next word prediction model in the style of GPT, built entirely from scratch. This project is all about building intuition and understanding of the fundamental concepts that underpin the Transformer architecture. The focus is not to be the fastest or most optimized model. Instead its aim is to be clear and educational for those wanting to understand the technical details of how transformers work.

To achieve this end the Transformer is broken down into its core components, each piece implemented separately.

---
#### Project Structure
The project is structured in a hierarchical and modular fashion according to the original "attention is all you need" paper (Vaswani et al., 2017). As such the code in the components folder contain most of the detail, while code such as the model.py contain the transformer which ties all the pieces together. The neural_net folder contains a mini neural network library complete with a grad engine which is really an extension of Andrej Karpathy's micrograd, for details i refer back to his brilliant material and repo at: https://github.com/karpathy/micrograd
