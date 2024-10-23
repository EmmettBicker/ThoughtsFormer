import torch
import torch.nn as nn
import torch.nn.functional as F

# A modified version of pytorch's positional encoding implementaiton
class DualPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, context_len: int = 512, max_thought_len: int = 4, disable_thought_encodings_if_thought_len_is_zero: bool = True, sinusoidal=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_embed = d_model
      
        max_len = context_len // (max_thought_len + 1)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.math.log(10000.0) / d_model))
        if sinusoidal == True:
          sinusoidal_positional_encoding = torch.zeros(1, max_len, d_model)
          sinusoidal_positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
          sinusoidal_positional_encoding[0, :, 1::2] = torch.cos(position * div_term)
          self.register_buffer('pe', sinusoidal_positional_encoding)
        else:
          self.learned_positional_encoding = nn.Embedding(max_len, self.d_embed)
        self.sinusoidal = sinusoidal
        self.max_len, self.max_thought_len, self.context_len = max_len, max_thought_len, context_len
        self.disable_thought_encoding = disable_thought_encodings_if_thought_len_is_zero and max_thought_len == 0

        if not self.disable_thought_encoding:
          self.position_in_thought_encoding = nn.Embedding(max_thought_len+1, self.d_embed)
          self.position_in_thought_encoding.weight.data.normal_(mean=0.0, std=0.01) # I need to test if this works 

    def forward(self, x: torch.Tensor, n_thoughts_taken: int) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # plt.imshow(x[0,:,:].detach().numpy())
        # plt.show()
        x = simple_batched_reshape_with_offset(x, max_seq_length=self.max_len, thoughts_taken=n_thoughts_taken)
        # print(x.shape)
        if self.sinusoidal:
          x = x + self.pe.unsqueeze(2)
        else:
          x = x + self.learned_positional_encoding(torch.arange(self.max_len).to(x.device)).unsqueeze(0).unsqueeze(2)
          
        if not self.disable_thought_encoding:
          x = x + self.position_in_thought_encoding(torch.arange( self.max_thought_len + 1).to(x.device)).unsqueeze(0).unsqueeze(0)
        # plt.title("X on the token positional encodings")
        # plt.imshow(x[0,:,0,:].detach().numpy())
        # plt.show()
        # plt.title("X on the token positional encodings")
        # plt.imshow(x[0,:,1,:].detach().numpy())
        # plt.show()
        # plt.title("X on the thought position encodings")
        # plt.imshow(x[0,0,:,:].detach().numpy())
        # plt.show()
        # Reshape and pad back to original size
        x = inverse_simple_batched_reshape_with_offset(x,thoughts_taken=n_thoughts_taken)

        # plt.imshow(x[0,:,:].detach().numpy())
        # plt.show()
        # print(x.shape)
        # padding_size = self.max_len - (real_token_count * n_thoughts_taken)
        # print(x.shape)
        # if padding_size > 0:
        #     x = F.pad(x, (0, 0, 0, padding_size), mode='constant', value=0)
        # print(x.shape)
        return self.dropout(x)


def simple_batched_reshape_with_offset(x: torch.Tensor, max_seq_length: int, thoughts_taken: int) -> torch.Tensor:
    thoughts = thoughts_taken + 1
    max_thoughts = x.size(1) // max_seq_length
    x = x[:,:max_seq_length*thoughts,:].view(x.size(0), max_seq_length, thoughts, x.size(2))
    return F.pad(x,(0,0, 0, (max_thoughts - thoughts)))


def inverse_simple_batched_reshape_with_offset(x: torch.Tensor, thoughts_taken: int) -> torch.Tensor:
  seq_len, max_thoughts = x.size(1), x.size(2)
  x = x[:,:,:thoughts_taken+1,:].reshape(x.size(0), -1 ,x.size(3))
  x = F.pad(x,(0,0,0, seq_len * (max_thoughts - (thoughts_taken+1))))
  return x