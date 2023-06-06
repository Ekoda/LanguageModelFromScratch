import numpy as np
from utils.generic_utils import load_text_data
from transformer.transformer import EssentialTransformer


data = load_text_data(path="data/test.txt")
model = EssentialTransformer(data, 64, 2, 2)

forward_pass = model.forward("the quick fox jumped", "over")

print(forward_pass.shape)