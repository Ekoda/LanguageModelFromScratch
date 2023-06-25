import re


def build_vocab(tokens: list[str]) -> dict[str, int]:
    sorted_unique_tokens = sorted(set(tokens))
    return {word: i for i, word in enumerate(sorted_unique_tokens)}

def tokenize(text: str) -> list[str]:
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    tokens = text.split()
    return tokens
