from transformers import GPT2Model
import torch.nn as nn
import torch
pretrained = GPT2Model.from_pretrained('gpt2')
for k, v in pretrained.named_parameters():
    if k == 'wte.weight':
        loaded_embeddings = v
        num_embeddings, embedding_dim = loaded_embeddings.shape
        word_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        word_embeddings.weight = nn.Parameter(loaded_embeddings)
        
def get_word_embeddings(x: torch.Tensor) -> torch.Tensor:
    return word_embeddings(x)