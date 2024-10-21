# Authored by Claude

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_sequence_length, num_layers=8, n_head=8, d_embed=768, feed_forward_dim=2048, dropout=0.1):
        super().__init__()
        self.d_embed = d_embed
        self.max_sequence_length = max_sequence_length

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_sequence_length, d_embed))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_embed, nhead=n_head, 
                                                   dim_feedforward=feed_forward_dim, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers for PPO
        self.policy_head = nn.Linear(d_embed, vocab_size)
        self.value_head = nn.Linear(d_embed, 1)

    def forward(self, x, padding_mask=None):
        # x shape: (batch_size, seq_len)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Get policy and value outputs
        policy_logits = self.policy_head(x)
        
        return policy_logits
    
    def forward_tokens(self, x, padding_mask=None):
        # x shape: (batch_size, seq_len)
        
        # Embed tokens
        x = self.token_embedding(x)
        return self.forward(x,padding_mask)
