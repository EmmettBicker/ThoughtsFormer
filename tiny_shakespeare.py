from datasets import load_dataset
from transformers import GPT2Tokenizer
from enum import Enum
import random
from torch.utils.data import Dataset
from character_tokenizer import ShakespeareCharacterTokenizer

class TokenizerType(Enum):
    GPT2 = 0
    CHARACTER_LEVEL = 1

class TinyShakespeareDataset(Dataset):
    def __init__(self, token_window_size, window_offset, split="train", tokenizer=TokenizerType.GPT2):
        self.token_window_size = token_window_size
        self.window_offset = window_offset
        text = load_dataset("tiny_shakespeare", split=split)['text'][0]
     
        assert isinstance(tokenizer, TokenizerType)
        if tokenizer == TokenizerType.GPT2:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokens = tokenizer.encode(text, return_tensors="pt")[0]
        elif tokenizer == TokenizerType.CHARACTER_LEVEL:
            tokenizer = ShakespeareCharacterTokenizer()
            self.tokens = tokenizer.encode(text)

    def __len__(self):
        return (len(self.tokens) - self.token_window_size) // self.window_offset + 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        
        start = idx * self.window_offset
        end = start + self.token_window_size + 1
        
        return self.tokens[start:end-1], self.tokens[start+1:end]
    
    def shuffle(self):
        random.shuffle(self.indices)