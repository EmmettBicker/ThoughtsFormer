import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from enum import Enum 
from typing import Type


MAX_SEQUENCE_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug_print(*args, flag=False, **kwargs):
    if flag:
        print(*args, **kwargs)

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

import math

def simple_batched_reshape_with_offset(x: torch.Tensor, max_seq_length: int, thoughts_taken: int) -> torch.Tensor:
    thoughts = thoughts_taken + 1
    max_thoughts = x.size(1) // max_seq_length
    x = x[:,:max_seq_length*thoughts,:].view(x.size(0), max_seq_length, thoughts, x.size(2))
    return F.pad(x,(0,0, 0, (max_thoughts - thoughts)))


def inverse_simple_batched_reshape_with_offset(x: torch.Tensor, thoughts_taken: int) -> torch.Tensor:
  seq_len, max_thoughts = x.size(1), x.size(2)
  x = x[:,:,:thoughts_taken+1,:].view(x.size(0), -1 ,x.size(3))
  x = F.pad(x,(0,0,0, seq_len * (max_thoughts - (thoughts_taken+1))))
  return x


# Manual implementation of attention because of weird inconsistencies in pytorch's built in version
class _NanAwareTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # print("Input to layer:", src)
        
        # Extract W_q, W_k, W_v, and biases
        with torch.no_grad():
            W = self.self_attn.in_proj_weight.to(torch.float64)
            b = self.self_attn.in_proj_bias.to(torch.float64)
            E = self.self_attn.embed_dim  # Total embedding size
            H = self.self_attn.num_heads  # Number of attention heads

            # Split weights for multi-head projections
            W_q, W_k, W_v = W[:E], W[E:2*E], W[2*E:]
            b_q, b_k, b_v = b[:E], b[E:2*E], b[2*E:]

        # Convert input to float64 for precision
        src_64 = src.to(torch.float64)
        batch_size, seq_len, embed_dim = src_64.size()

        # Compute Q, K, V with multiple heads
        def reshape_for_heads(x):
            """Reshape to (batch_size, num_heads, seq_len, head_dim)."""
            return x.view(batch_size, seq_len, H, embed_dim // H).transpose(1, 2)

        q = reshape_for_heads(torch.matmul(src_64, W_q.T) + b_q)
        k = reshape_for_heads(torch.matmul(src_64, W_k.T) + b_k)
        v = reshape_for_heads(torch.matmul(src_64, W_v.T) + b_v)

        # print("Q:", q)
        # print("K:", k)
        # print("V:", v)

        # Compute scaled dot-product attention for each head: (QK^T) / sqrt(d_k)
        d_k = q.size(-1)  # Head dimension
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float64))

        # Apply the optional mask
        if src_mask is not None:
            attn_scores += src_mask.to(torch.float64)

        # print("Pre-softmax attention scores:", attn_scores)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        # print("Post-softmax attention weights:", attn_weights)

        # Multiply attention weights with V
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        # print("Attention output (before reshaping):", attn_output)

        # Reshape back to (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        # print("Attention output (after reshaping):", attn_output)

        # Residual connection and dropout
        src = src + self.dropout1(attn_output.to(src.dtype))
        # print("After dropout and residual connection:", src)

        # Apply layer normalization
        src = self.norm1(src)
        # print("After first layer norm:", src)

        # Feedforward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # print("After feed-forward network:", src2)

        # Another residual connection and dropout
        src = src + self.dropout2(src2)
        # print("After dropout and second residual connection:", src)

        # Final layer normalization
        src = self.norm2(src)
        # print("After second layer norm:", src)

        # Handle NaN values by replacing them with 0
        output = torch.nan_to_num(src, nan=0.0)
        # print("Final output after NaN handling:", output)

        return output

# Normal version that's much faster but doesn't have the same theoretical soundness as previous version
class _FasterNanAwareTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # print("before: ", args[0])
        output = super().forward(*args, **kwargs)
        # print("after: ", output)
        return torch.where(torch.isnan(output), torch.zeros_like(output), output)

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class CausalTransformer(nn.Module):
    def __init__(self,
                 max_thought_len, 
                 max_sequence_length, 
                 num_layers, 
                 sinusoidal_position_encoding: bool = True,
                 d_embed=768, 
                 n_head=8, 
                 dim_feedforward=2048, 
                 dropout=0.1, 
                 activation=F.gelu, # Different from usual default
                 layer_norm_eps: float = 0.0001,
                 batch_first: bool = True, # Different from usual default
                 norm_first: bool = True, # Different from usual default
                 bias: bool = True,
                 device = None,
                 dtype = None,
                 debug: bool = False
                ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.debug = debug

        self.dual_positional_encoding = DualPositionalEncoding(d_embed, dropout, max_sequence_length, max_thought_len,sinusoidal=sinusoidal_position_encoding)
        if device is not None:
          raise NotImplemented("Device parameter being specified has not been implemented. Please use .to() on this module.")
        # device = device
        layer = _FasterNanAwareTransformerEncoderLayer(
            d_model=d_embed,
            nhead = n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            # device = device  || not implemented because I don't know if it will it in with the rest of my calculations
            dtype=dtype
        )
        # nn.TransformerEncoderLayer())
    
        self.transformer = nn.TransformerEncoder(encoder_layer=layer,
                                                 num_layers=num_layers,
                                                 norm=nn.LayerNorm(d_embed))


    def generate_thoughtsformer_mask(self, thoughts_taken, real_tokens):
      assert real_tokens <= self.max_sequence_length, f"Number of real tokens suggested by padding mask is too large. Padding mask shoukd be of size (batch_size x n_real_tokens). Recieved {real_tokens} tokens in padding mask when maximum is {self.max_sequence_length}"
      main_size = self.max_sequence_length
      block_size = thoughts_taken + 1
      n_tokens = real_tokens
      
      # Create the main tensor and block tensor
      causal_mask = torch.zeros((main_size, main_size))
      block_for_thought_sequence = torch.triu(torch.ones(block_size,block_size),diagonal=0)

      # List of starting indices for the diagonal blocks
      block_starting_idxs = torch.arange(n_tokens) * block_size

      for idx in block_starting_idxs:
          causal_mask[idx:idx+block_size, idx:idx+block_size] = block_for_thought_sequence
          causal_mask[idx, idx+1:n_tokens*block_size] = 1
  
      causal_mask = causal_mask.T == 0

      return causal_mask

    def generate_normal_causal_mask(self, *args):
      return torch.triu(torch.ones(self.max_sequence_length, self.max_sequence_length),diagonal=1)

    def forward(self, x, padding_mask, thoughts_taken, real_token_count):
      debug_print("embeddings right before positional encodings", x, flag=self.debug)
      x = self.dual_positional_encoding(x, thoughts_taken)
      torch.save(x,"test.pt")
      causal_mask = self.generate_thoughtsformer_mask(thoughts_taken,real_token_count).to(x.device)
      debug_print("embeddings right before transformer forward", x, flag=self.debug)
      if self.debug == True:
        plt.imshow(causal_mask)
        plt.show()
      x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
      debug_print("embeddings right after transformer forward", x, flag=self.debug)
      return x




class ThoughtsFormer(nn.Module):
  def __init__(self, max_thought_len, vocab_size, max_context_length=None,max_sequence_length=None, sinusoidal_position_encoding=True,num_layers=8, n_head=8, d_embed=768, dim_feedforward=2048, dropout=0.1, activation=F.gelu, layer_norm_eps=0.0001, batch_first=True, norm_first=True, bias=True, device=None, dtype=None, verbose=False):
      super().__init__()
      
      if max_context_length is None and max_sequence_length is None:
        raise ValueError("Must specify either max_context_length or max_sequence_length")
      elif max_context_length is None: # and max_context_length is not None
        max_context_length = max_sequence_length * (max_thought_len + 1) # If max_context_length is 20, and one thought is allowed, then it should be 
      
      if max_context_length is None:
        raise RuntimeError("Max context length undefined--unreachable code reached")
      if device is not None:
          raise NotImplemented("Device parameter being specified has not been implemented. Please use .to() on this module.")

      self.max_context_length, self.d_embed = max_context_length, d_embed
      self.max_thought_length = max_thought_len
      self.transformer = CausalTransformer(
        max_thought_len=max_thought_len, 
        max_sequence_length=max_context_length, 
        num_layers=num_layers, 
        sinusoidal_position_encoding=sinusoidal_position_encoding, 
        d_embed=d_embed,
        n_head=n_head, 
        dim_feedforward=dim_feedforward, 
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        batch_first=batch_first, # Different from usual default
        norm_first=norm_first, # Different from usual default
        bias=bias,
        device=device,
        dtype=dtype,
        debug=verbose
        )
      
      
      # self.policy_feedforward = nn.Sequential(
      #   nn.Linear(d_embed, d_embed),
      #   nn.GELU(),
      #   nn.Linear(d_embed, vocab_size)
      # )
      
      self.policy_feedforward = nn.Linear(d_embed, vocab_size,bias=False)
      # Weird difference between the two but GPT2 uses the single head for the feedforward head and I want to do the weight transfer entirely but allow extra modularity for the value function
      self.value_feedforward = nn.Sequential(
        nn.Linear(d_embed, dim_feedforward),
        nn.GELU(),
        nn.Linear(dim_feedforward, 1)
      )
      self.token_embedding = nn.Embedding(vocab_size, d_embed)
      
      self.debug = verbose

  
  def forward_ppo(self, state_embeddings: torch.Tensor, padding_mask: torch.Tensor, n_thoughts_taken: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Takes in the current state and the padding mask, outputs the logits for the next action and value for every token in the sequence
    '''
    if type(n_thoughts_taken) == torch.Tensor and n_thoughts_taken.numel() == 1:
      n_thoughts_taken: int  = n_thoughts_taken.item()
       
    assert n_thoughts_taken <= self.max_thought_length, f"n_thoughts_taken ({n_thoughts_taken}) > thought_len ({self.max_thought_length})"
    assert state_embeddings.ndim == 3, f"state_embeddings must be three dimensional, where the dimensons are (batch_size, seq_len, ndim). Instead recieved {state_embeddings.ndim} dimensional input"
    assert state_embeddings.size(2) == self.d_embed, f"state_embedding's final dimension must be of size d_embed ({self.d_embed}), instead recieved size ({state_embeddings.size(2)})"
    
    assert state_embeddings.size(0) == padding_mask.size(0), f"mismatch between the size of dimension 0 in state_embedding ({state_embeddings.size(0)}) and the size of dimension 0 in padding_mask ({padding_mask.size(0)}). This is the batch size so these dimensions should be equal values"
    
    state_embeddings = self.prepare_thoughtsformer_embedding_input(state_embeddings)
    
    padding_mask = self.prepare_thoughtsformer_padding_mask_input(padding_mask)
    
    self.token_positions = torch.where(padding_mask == False)[1]
    # print(torch.sum(padding_mask == 0,dim=1))
    # print( int(torch.sum(padding_mask == 0,dim=1).max()))
    self.n_real_tokens = int(torch.sum(padding_mask == 0,dim=1).max())
    debug_print("original embeddings \n", state_embeddings, flag=self.debug)
    next_embeddings = self.transformer(state_embeddings, padding_mask, n_thoughts_taken, self.n_real_tokens)
    debug_print("future embeddings\n", next_embeddings,  flag=self.debug)
    action_embeddings = self.get_tokens_at_action_location(next_embeddings, n_thoughts_taken)
    
    return self.policy_feedforward(action_embeddings), self.value_feedforward(action_embeddings).squeeze(dim=2)
    
  # Claude's version for tokens!
  def forward_ppo_with_tokens(self, tokens: torch.Tensor, padding_mask: torch.Tensor, n_thoughts_taken: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Accepts raw tokens, embeds them, and then calls the existing forward_ppo method.
    
    Args:
    tokens (torch.Tensor): Input tokens of shape (batch_size, seq_len)
    padding_mask (torch.Tensor): Padding mask of shape (batch_size, seq_len)
    n_thoughts_taken (int): Number of thoughts taken
    
    Returns:
    Tuple of action logits and value estimates
    """
    # Assert input dimensions
    assert tokens.ndim == 2, f"tokens must be two-dimensional (batch_size, seq_len). Received shape: {tokens.shape}"
    assert tokens.shape == padding_mask.shape, f"tokens and padding_mask must have the same shape. Got {tokens.shape} and {padding_mask.shape}"
    
    if padding_mask.dtype != torch.bool:
      padding_mask = padding_mask.bool()
    # Embed the tokens
    state_embeddings = self.token_embedding(tokens)
    
    # Call the existing forward_ppo method
    return self.forward_ppo(state_embeddings, padding_mask, n_thoughts_taken)
  # def forward(self, state_embeddings, padding_mask) -> tuple[torch.Tensor, torch.Tensor]:

  #   # Embeddings should be size N x seq_len x d_embed
  #   self.token_positions = torch.where(padding_mask == False)[1]
  #   self.n_real_tokens = int(self.max_context_length - torch.sum(padding_mask))

  #   original_embeddings = state_embeddings
  #   debug_print("original embeddings \n", state_embeddings, flag=self.debug)

  #   for thoughts_taken in range(self.thought_length+1):
  #     debug_print(f"initiating loop for thought {thoughts_taken}", flag=self.debug)
  #     next_embeddings = self.transformer(state_embeddings, padding_mask, thoughts_taken, self.n_real_tokens)
  #     # next_embeddings = torch.arange(self.seq_len).view(1,-1,1).repeat([embeddings.size(0),1,d_embed])
  #     debug_print(next_embeddings, next_embeddings.shape, flag=self.debug)
  #     if thoughts_taken != self.thought_length: # Don't need to insert next thoughts if there's not going to be another iteration
  #       state_embeddings = self.insert_thoughts(next_embeddings, original_embeddings, padding_mask, thoughts_taken + 1)
  #     debug_print("updated embeddings\n", state_embeddings,  flag=self.debug)

  #     original_embeddings = state_embeddings

  #   return self.policy_feedforward(state_embeddings), self.value_feedforward(state_embeddings)
    
  def prepare_thoughtsformer_embedding_input(self, x: torch.Tensor):
    # Expected shape batch_size, seq_len, d_embed
    # Want to zero pad the seq_len dimension to be max_seq_len + max_seq_len * thought_length
    max_context_length = self.max_context_length
    
    seq_len = x.size(1) 
    assert seq_len <= max_context_length, f"Length of the input embeddings ({seq_len}) exceeds maximum context length ({max_context_length}). Sequence length is dimension 1 of the input embeddings."
    
    return F.pad(x, (0,0,0,max_context_length - seq_len))

  def prepare_thoughtsformer_padding_mask_input(self, padding_mask: torch.Tensor):
      # Expected shape batch_size, seq_len
      # Want to zero pad the seq_len dimension to be max_seq_len + max_seq_len * thought_length
      max_context_length = self.max_context_length
    
      seq_len = padding_mask.size(1) 
      assert seq_len <= max_context_length, f"Length of the padding mask's ({seq_len}) exceeds maximum context length ({max_context_length}). Sequence length is dimension 1 of the input embeddings."
      
      return F.pad(padding_mask, (0,max_context_length - seq_len), value=1)
      
    
  def insert_thoughts(self, next_embeddings, original_embeddings, padding_mask, iter):

    debug_print("Debugging here ", self.n_real_tokens, self.token_positions.size(0), flag=self.debug)
    n_elements = self.token_positions.size(0) * iter
    n_element_next = self.token_positions.size(0) * (iter + 1)
    batch_size, seq_len, d_embed = original_embeddings.shape
    # we'll reshape and concat
    # to go from
    # 1, t            # 1, t, t
    # 2, t    --->    # 2, t, t
    # 3, t.flatten()  # 3, t, t.flatten()
    
    # 1 x nrealtokens --> n x n_thoughts
    # could do 1 x n_tokens --> n x n_max_thoughts

    original_embeddings = original_embeddings[:,:n_elements,:].view(batch_size,-1, iter, d_embed)


    # This gets the positions of the next tokens to predict - 1, so right before the tokens that are being predicted
    next_token_positions = (torch.arange(self.token_positions.size(0)) + 1) * iter - 1
    next_embeddings = next_embeddings[:, next_token_positions,:]

    # Reshapes the embeddings so they can be concatenated like in the previous diagram
    next_embeddings = next_embeddings.view(next_embeddings.size(0), next_embeddings.size(1), 1, next_embeddings.size(2))

    #Concatenates and reshapes back
    final_embeds = torch.cat((original_embeddings,next_embeddings),dim=2)
    final_embeds = final_embeds.view(batch_size,-1,d_embed)
    debug_print("final embedding shape", flag=self.debug)
    debug_print(final_embeds.shape, seq_len, n_element_next, flag=self.debug)
    padding = torch.zeros(final_embeds.size(0), seq_len-n_element_next, final_embeds.size(2))
    final_embeds = torch.cat((final_embeds, padding),dim=1)

    self.token_positions = self.get_next_token_count(self.token_positions)

    return final_embeds

  def get_tokens_at_action_location(self, embeddings_or_logits, thought_length):
    n_real_tokens = self.max_context_length // (self.max_thought_length + 1)
    token_predictor_locations = (torch.arange(n_real_tokens) + 1) * (thought_length + 1) - 1
    # print(logits)
    # print(token_predictor_locations)
    return embeddings_or_logits[:,token_predictor_locations,:]
    # each token_predictor is the end result of the previous n_thoughts tokens, so if it's 1 thought, there's one behind it (the original handler of the original token)
    # token_logits = self.out(embeddings[:,token_predictor_locations]) # shape: batch x token_predictor_count x vocab_size

  def prepare_thoughtsformer_input(self, x: torch.Tensor, max_seq_len: int, max_thought_length: int):
    # Expected shape batch_size, seq_len, d_embed
    # Want to zero pad the seq_len dimension to be max_seq_len + max_seq_len * thought_length
    
    max_context_length = max_seq_len * (max_thought_length + 1)
    seq_len = x.size(1) 
    
    return F.pad(x, (0,0,0,max_context_length - seq_len))

  def prepare_thoughtsformer_padding_mask(self, padding_mask: torch.Tensor, max_seq_len: int, max_thought_length: int):
      # Expected shape batch_size, seq_len
      # Want to zero pad the seq_len dimension to be max_seq_len + max_seq_len * thought_length
      
      max_context_length = max_seq_len * (max_thought_length + 1)
      seq_len = padding_mask.size(1) 
      
      return F.pad(padding_mask, (0,max_context_length - seq_len), value=1)
  def get_next_token_count(self, token_positions):
    '''
    Updates the internal token_positions variable. Assumes each thought train will have the same length.
    '''
    return token_positions + torch.arange(self.token_positions.shape[0])
  
  @classmethod
  def from_pretrained_GPT2(cls: Type['ThoughtsFormer'], n_thoughts: int = 0) -> 'ThoughtsFormer':
    from transformers import GPT2LMHeadModel
    pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
    
    max_sequence_length = 1024//(n_thoughts+1)
    thoughtsformer = cls(
      max_thought_len=n_thoughts,
      vocab_size=50257,
      max_context_length=1024,
      sinusoidal_position_encoding=False,
      num_layers=12,
      n_head=12,
      d_embed=768,
      dim_feedforward=768 * 4,
      dropout=0.1,
      activation=NewGELUActivation(),
      norm_first=True, # this is default behavior but just to be safe
    )
    p_dict = pretrained.state_dict()
    t_dict = thoughtsformer.state_dict()

    explicit_map = {
        'token_embedding.weight' : 'transformer.wte.weight',
        'policy_feedforward.weight' : 'lm_head.weight',
        'transformer.transformer.norm.weight' : 'transformer.ln_f.weight',
        'transformer.transformer.norm.bias' : 'transformer.ln_f.bias',
        'transformer.dual_positional_encoding.learned_positional_encoding.weight' : 'transformer.wpe.weight'
    }

    for k, v in explicit_map.items():
        # print(k,v)
        # print(t_dict[k].shape, p_dict[v].shape)
        if v == "transformer.wpe.weight":
          assert t_dict[k].shape == p_dict[v][0:max_sequence_length].shape
          t_dict[k].copy_(p_dict[v][0:max_sequence_length])
        else:
          assert t_dict[k].shape == p_dict[v].shape
          t_dict[k].copy_(p_dict[v])
        
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
                assert p_dict[v].shape[::-1] == t_dict[k].shape
                with torch.no_grad():
                    t_dict[k].copy_(p_dict[v].t())
            else:
                with torch.no_grad():
                    assert t_dict[k].shape == p_dict[v].shape
                    t_dict[k].copy_(p_dict[v])
        
    thoughtsformer.load_state_dict(t_dict)
    return thoughtsformer
    