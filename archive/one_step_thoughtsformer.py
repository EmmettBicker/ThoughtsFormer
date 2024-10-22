import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from enum import Enum 

MAX_SEQUENCE_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug_print(*args, flag=False, **kwargs):
    if flag:
        print(*args, **kwargs)

# A modified version of pytorch's positional encoding implementaiton
class DualPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512, max_thought_len: int = 4):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_embed = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)


        self.register_buffer('pe', pe)
        self.max_len, self.max_thought_len = max_len, max_thought_len
        self.thought_position_encoding = nn.Embedding(max_thought_len+1, self.d_embed)

    def forward(self, x: torch.Tensor, n_thoughts_taken: int, real_token_count: int) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        
        batch_size = x.size(0)
        n_thoughts_taken = n_thoughts_taken + 1

        # Reshape to put thoughts per token taken on the same dimension
        x = x[:,:real_token_count * n_thoughts_taken,:].view(batch_size,real_token_count, n_thoughts_taken, self.d_embed)
        # Add both kinds of embeddings
        x = x + self.pe[:,:real_token_count].unsqueeze(2)
        x = x + self.thought_position_encoding(torch.arange(n_thoughts_taken).to(x.device)).unsqueeze(0).unsqueeze(0)

        # Reshape and pad back to original size
        x = x.view(batch_size, -1, self.d_embed)

        padding_size = self.max_len - (real_token_count * n_thoughts_taken)
        if padding_size > 0:
            x = F.pad(x, (0, 0, 0, padding_size), mode='constant', value=0)

        return self.dropout(x)


# This is a bad solution to this issue, but for development I like seeing the clearly nan or 0 values.
# class NanAwareTransformerEncoderLayer(nn.TransformerEncoderLayer):
#     def forward(self, src, *args, **kwargs):
#         # Call the original forward method with all arguments
#         output = super().forward(src, *args, **kwargs)

#         # Zero out NaNs
#         print("loop :)", output)
#         output = torch.nan_to_num(output, nan=0.0)
#         print(output)
        
#         return output


import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Currently unused version that's much faster but doesn't have the same theoretical soundness as previous version
class _FasterNanAwareTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        return torch.where(torch.isnan(output), torch.zeros_like(output), output)

class CausalTransformer(nn.Module):
    def __init__(self,max_thought_len, max_sequence_length, num_layers, n_head=8, d_embed=768, feed_forward_dim=2048, dropout=0.1, debug=False):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.debug = debug

        self.positional_encoding = DualPositionalEncoding(d_embed, dropout, max_sequence_length, max_thought_len)
        self.layer = _NanAwareTransformerEncoderLayer(
            d_model=d_embed,
            nhead = n_head,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=self.layer,
                                                 num_layers=num_layers,
                                                 norm=nn.LayerNorm(d_embed))


    def generate_thoughtsformer_mask(self, thoughts_taken, real_tokens):

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
      x = self.positional_encoding(x, thoughts_taken, real_token_count)
      causal_mask = self.generate_thoughtsformer_mask(thoughts_taken,real_token_count).to(x.device)
      debug_print("embeddings right before transformer forward", x, flag=self.debug)
      if self.debug == True:
        plt.imshow(causal_mask)
        plt.show()
      x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
      debug_print("embeddings right after transformer forward", x, flag=self.debug)
      return x




class ThoughtsFormer(nn.Module):
    def __init__(self, thought_length, vocab_size, max_context_length=None,max_sequence_length=None, num_layers=8, n_head=8, d_embed=768, feed_forward_dim=2048, dropout=0.1, verbose=False):
        super().__init__()
        
        if max_context_length is None and max_sequence_length is None:
          raise ValueError("Must specify either max_context_length or max_sequence_length")
        elif max_context_length is None: # and max_context_length is not None
          max_context_length = max_sequence_length * (thought_length + 1) # If max_context_length is 20, and one thought is allowed, then it should be 
        
        if max_context_length is None:
          raise RuntimeError("Max context length undefined--unreachable code reached")

        self.max_context_length, self.d_embed = max_context_length, d_embed
        self.thought_length = thought_length
        self.transformer = CausalTransformer(thought_length, max_context_length, num_layers, n_head, d_embed, feed_forward_dim, dropout, debug=verbose)
        
        self.policy_feedforward = nn.Sequential(
          nn.Linear(d_embed, feed_forward_dim),
          nn.GELU(),
          nn.Linear(feed_forward_dim, vocab_size)
        )
        
        self.value_feedforward = nn.Sequential(
          nn.Linear(d_embed, feed_forward_dim),
          nn.GELU(),
          nn.Linear(feed_forward_dim, 1)
        )
        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        
        self.debug = verbose

    
    def forward_ppo(self, state_embeddings: torch.Tensor, padding_mask: torch.Tensor, n_thoughts_taken: int) -> tuple[torch.Tensor, torch.Tensor]:
      '''
      Takes in the current state and the padding mask, outputs the logits for the next action and value for every token in the sequence
      '''
      assert n_thoughts_taken <= self.thought_length, f"n_thoughts_taken ({n_thoughts_taken}) > thought_len ({self.thought_length})"
      assert state_embeddings.ndim == 3, f"state_embeddings must be three dimensional, where the dimensons are (batch_size, seq_len, ndim). Instead recieved {state_embeddings.ndim} dimensional input"
      assert state_embeddings.size(2) == self.d_embed, f"state_embedding's final dimension must be of size d_embed ({self.d_embed}), instead recieved size ({state_embeddings.size(2)})"
      
      assert state_embeddings.size(0) == padding_mask.size(0), f"mismatch between the size of dimension 0 in state_embedding ({state_embeddings.size(0)}) and the size of dimension 0 in padding_mask ({padding_mask.size(0)}). This is the batch size so these dimensions should be equal values"
      
      state_embeddings = self.prepare_thoughtsformer_embedding_input(state_embeddings)
      padding_mask = self.prepare_thoughtsformer_padding_mask_input(padding_mask)
      
      self.token_positions = torch.where(padding_mask == False)[1]
      self.n_real_tokens = int(torch.sum(padding_mask == 0))
      debug_print("original embeddings \n", state_embeddings, flag=self.debug)
      next_embeddings = self.transformer(state_embeddings, padding_mask, n_thoughts_taken, self.n_real_tokens)
      debug_print("future embeddings\n", next_embeddings,  flag=self.debug)
      action_embeddings = self.get_tokens_at_action_location(next_embeddings, n_thoughts_taken)
      
      return self.policy_feedforward(action_embeddings), self.value_feedforward(action_embeddings)
      
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
      n_real_tokens = self.n_real_tokens
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