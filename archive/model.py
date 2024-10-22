import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

MAX_SEQUENCE_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pytorch's   positional encoding implementaiton
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class CausalTransformer(nn.Module):
    def __init__(self, max_sequence_length, num_layers, n_head=8, d_embed=768, feed_forward_dim=2048, dropout=0.1):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead = n_head,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=self.layer,
                                                 num_layers=num_layers,
                                                 norm=nn.LayerNorm(d_embed))
        
        
        
    def forward(self, x, padding_mask):
        causal_mask = torch.triu(torch.ones(self.max_sequence_length, self.max_sequence_length), diagonal=1)
        causal_mask = causal_mask.bool().to(x.device)
        return self.transformer.forward(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)
        
class CausalModel(nn.Module):
    def __init__(self, vocab_size, max_sequence_length, num_layers, n_head=8, d_embed=768, feed_forward_dim=2048, dropout=0.1):
        super().__init__()
        self.transformer = CausalTransformer(max_sequence_length, num_layers, n_head, d_embed, feed_forward_dim, dropout)
        self.out = nn.Linear(d_embed, vocab_size)
        
    def forward(self, x, padding_mask):
        x = self.transformer(x, padding_mask)
        x = self.out(x)
        return x
