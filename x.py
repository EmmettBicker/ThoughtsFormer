from tiny_shakespeare import TinyShakespeareDataset
import os 
import torch


x = TinyShakespeareDataset(260//5,51200)

from thoughtsformer import ThoughtsFormer
m = ThoughtsFormer(max_context_length=260, max_thought_len=4,vocab_size=50257,num_layers=1).to('cuda')
# m = ThoughtsFormer.from_pretrained_GPT2(1,reinforcement_learning=False).to('cuda')
tokens = torch.stack((x[0][0],x[1][0])).to('cuda')
import torch
m(tokens, torch.zeros_like(tokens).bool())
print("done :)")