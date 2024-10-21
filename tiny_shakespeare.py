from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset

class TinyShakespeareDataset(Dataset):
    def __init__(self, token_window_size, window_offset):
        self.token_window_size = token_window_size
        self.window_offset = window_offset
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        text = load_dataset("tiny_shakespeare", split="train")['text'][0]
        self.tokens = tokenizer.encode(text, return_tensors="pt")[0]

    def __len__(self):
        return (len(self.tokens) - self.token_window_size) // self.window_offset + 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        
        start = idx * self.window_offset
        end = start + self.token_window_size + 1
        
        return self.tokens[start:end-1], self.tokens[start+1:end]