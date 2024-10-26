import torch
from datasets import load_dataset

class ShakespeareCharacterTokenizer():
    def __init__(self):
        # Extracting text
        dataset = load_dataset("tiny_shakespeare", split=None)
        full_text = dataset['train']['text'][0] + dataset['test']['text'][0] + dataset['validation']['text'][0]
        chars = sorted(list(set(full_text)))
        vocab_size = len(chars)

        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
    def encode(self, s):
        return torch.tensor([self.stoi[c] for c in s]) # encoder: take a string, output a list of integers
    def decode(self, l):
        if type(l) == torch.Tensor:
            return ''.join([self.itos[i.item()] for i in l]) # decoder: take a list of integers, output a string
        else:
            return ''.join([self.itos[i] for i in l])