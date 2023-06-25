import pytest

from src.utils.data_utils import *


def test_sequence_even_data():
    tokens = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
    sequences = sequence_data(tokens, sequence_length=3)
    assert sequences == [["the", "quick", "brown"], ["fox", "jumped", "over"], ["the", "lazy", "dog"]]

def test_sequence_odd_data():
    tokens = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog", "again"]
    sequences = sequence_data(tokens, sequence_length=3)
    assert sequences == [["the", "quick", "brown"], ["fox", "jumped", "over"], ["the", "lazy", "dog"], ["again"]]

def test_sequence_single_data():
    tokens = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
    sequences = sequence_data(tokens, sequence_length=1)
    assert sequences == [["the"], ["quick"], ["brown"], ["fox"], ["jumped"], ["over"], ["the"], ["lazy"], ["dog"]]

def test_sequence_empty_data():
    tokens = []
    sequences = sequence_data(tokens, sequence_length=1)
    assert sequences == []
