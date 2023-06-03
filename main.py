import numpy as np
from utils.generic_utils import load_text_data
from transformer.transformer import EssentialTransformer

data = load_text_data(path="data/test.txt")
model = EssentialTransformer(data)

forward_pass = model.forward("quick fox", "jumped")

print(forward_pass)