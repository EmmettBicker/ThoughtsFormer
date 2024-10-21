import torch
input = torch.load("test.pt", weights_only=True)

import torch
from transformers import GPT2LMHeadModel
from thoughtsformer import ThoughtsFormer
# t = GPT2Tokenizer.from_pretrained("gpt2")
pret = GPT2LMHeadModel.from_pretrained("gpt2")
custom_model = ThoughtsFormer(num_layers=12,vocab_size=50257, d_embed = 768, dim_feed_forward=768 * 4, n_head=12, max_context_length=1024, thought_length=0, dropout=0.15, sinusoidal_position_encoding=False)
pdict = pret.state_dict()
cdict = custom_model.state_dict()
# sentence = "Once upon a time"
pdict = pret.state_dict()
cdict = custom_model.state_dict()

explicit_map = {
    'token_embedding.weight' : 'transformer.wte.weight',
    'policy_feedforward.weight' : 'lm_head.weight',
    'transformer.transformer.norm.weight' : 'transformer.ln_f.weight',
    'transformer.transformer.norm.bias' : 'transformer.ln_f.bias',
    'transformer.dual_positional_encoding.learned_positional_encoding.weight' : 'transformer.wpe.weight'
}

param_counter = 0

for k, v in explicit_map.items():
    param_counter+= 1
    assert cdict[k].shape == pdict[v].shape
    cdict[k].copy_(pdict[v])
    
for i in range(12):
    layer_explicit_map = {
        f"transformer.transformer.layers.{i}.linear1.weight" : f"transformer.h.{i}.mlp.c_fc.weight" ,
        f"transformer.transformer.layers.{i}.linear1.bias" : f"transformer.h.{i}.mlp.c_fc.bias" ,
        f"transformer.transformer.layers.{i}.linear2.weight" : f"transformer.h.{i}.mlp.c_proj.weight", 
        f"transformer.transformer.layers.{i}.linear2.bias" : f"transformer.h.{i}.mlp.c_proj.bias" ,
        f"transformer.transformer.layers.{i}.norm1.weight" : f"transformer.h.{i}.ln_1.weight", 
        f"transformer.transformer.layers.{i}.norm1.bias" : f"transformer.h.{i}.ln_1.bias", 
        f"transformer.transformer.layers.{i}.norm2.weight" : f"transformer.h.{i}.ln_2.weight", 
        f"transformer.transformer.layers.{i}.norm2.bias" : f"transformer.h.{i}.ln_2.bias", 
        f"transformer.transformer.layers.{i}.self_attn.out_proj.weight" : f"transformer.h.{i}.attn.c_proj.weight", 
        f"transformer.transformer.layers.{i}.self_attn.out_proj.bias" : f"transformer.h.{i}.attn.c_proj.bias",
        f"transformer.transformer.layers.{i}.self_attn.in_proj_weight":  f"transformer.h.{i}.attn.c_attn.weight",
        f"transformer.transformer.layers.{i}.self_attn.in_proj_bias" : f"transformer.h.{i}.attn.c_attn.bias"
    }
    
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    for k, v in layer_explicit_map.items():
        
        if any(v.endswith(w) for w in transposed):
            assert pdict[v].shape[::-1] == cdict[k].shape
            with torch.no_grad():
                cdict[k].copy_(pdict[v].t())
        else:
            # print(cdict[k].shape, pdict[v].shape, k, v)
            with torch.no_grad():
                assert cdict[k].shape == pdict[v].shape
                cdict[k].copy_(pdict[v])
    
custom_model.load_state_dict(cdict)
# cdict['token_embedding.weight'].shape
# cdict['transformer.transformer.layers.0.linear2.weight'].shape

sentence = "once upon a time"


from bpe import BPETokenizer
tokenizer = BPETokenizer()

# print("\n\nDesired Output")
# print(pret(tokens,output_hidden_states=True).hidden_states[1])


# Desired Output
# tensor([[[ 0.3191,  0.6466,  1.1327,  ..., -0.0743,  0.7379, -0.3372],
#          [-0.3657, -0.3845, -0.2498,  ..., -0.0071,  0.2876,  0.2585],
#          [-1.4182,  0.1991, -0.4063,  ...,  0.0237,  0.1473, -0.4916],
#          [ 0.2380, -0.8441, -0.0074,  ...,  0.6175, -0.6014,  0.4939]]],
#        grad_fn=<AddBackward0>)

import torch
import torch.nn as nn
encoder_layer: nn.TransformerEncoderLayer = custom_model.transformer.transformer.layers[0]
mask = torch.zeros_like(input[:,:,0]).bool()
encoder_layer.eval()
out = encoder_layer.forward(input, src_mask=mask, is_causal=True)
print(out)