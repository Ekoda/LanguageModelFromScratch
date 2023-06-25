import re
import pytest

from src.preprocessing.tokenization import tokenize


def test_tokenize():
    assert tokenize("Hello, world!") == ['hello', 'world']
    assert tokenize("Multiple   spaces") == ['multiple', 'spaces']
    assert tokenize("Punctuation, marks; affect-tokenization!") == ['punctuation', 'marks', 'affect', 'tokenization']
    assert tokenize("") == []
    assert tokenize("ALL CAPS") == ['all', 'caps']

def test_tokenize_special_chars():
    assert tokenize("!@# $%^ &*()+") == []

def test_tokenize_with_numbers():
    assert tokenize("12345") == ['12345']
    assert tokenize("123 abcd 456 efg") == ['123', 'abcd', '456', 'efg']

def test_tokenize_different_languages():
    assert tokenize("Hola, mundo!") == ['hola', 'mundo']
    assert tokenize("Привет, мир!") == ['привет', 'мир']
    assert tokenize("你好，世界！") == ['你好', '世界']