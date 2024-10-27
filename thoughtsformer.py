import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import matplotlib.pyplot as plt
from typing import Type, Callable
from dual_positional_encoding import DualPositionalEncoding

MAX_SEQUENCE_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug_print(*args, flag=False, **kwargs):
    if flag:
        print(*args, **kwargs)


class _NanAwareTransformerEncoderLayer(nn.TransformerEncoderLayer):
  '''
  Implentation of <code>nn.TransformerEncoderLayer</code> that zeros all nans
  after execution. Nans should not be encountered in training when causal mask
  includes padding tokens.
  '''
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
    
    Copied from huggingface transformers library
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class ThoughtCausalTransformer(nn.Module):
  '''
  
  '''
  def __init__(self,
                max_thought_len, 
                max_context_length, 
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
    self.max_context_length = max_context_length
    self.max_sequence_length = max_context_length // (max_thought_len + 1)
    self.debug = debug

    self.dual_positional_encoding = DualPositionalEncoding(d_embed, dropout, max_context_length, max_thought_len,sinusoidal=sinusoidal_position_encoding)
    if device is not None:
      raise NotImplemented("Device parameter being specified has not been implemented. Please use .to() on this module.")
    # device = device
    layer = _NanAwareTransformerEncoderLayer(
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


  def generate_thoughtsformer_mask(self, thoughts_taken: int) -> torch.Tensor:
    '''
    Generates a causal mask where thoughts are invisible to all other tokens 
    except for tokens later in the same train of thought.
    '''
    sz = self.max_context_length
    t = thoughts_taken + 1
    def create_vertical_lines(size, spacing):
      indices = torch.arange(size)
      pattern = (indices % spacing != 0).float()
      lines = pattern.expand(size, -1)
      return lines

    lines= create_vertical_lines(sz,t).bool()
    blocks = ~torch.block_diag(*torch.ones(sz//t+1,t,t)).bool()[0:sz,0:sz]
    line_blocks = torch.bitwise_and(lines,blocks)
    triu = torch.triu(torch.ones(sz,sz), diagonal=1).bool()
    mask = torch.bitwise_or(line_blocks,triu)

    return mask

  def forward(self, x, padding_mask, thoughts_taken):
    '''
    Forward method of a thoughtsformer that makes thoughts invisible to all
    other tokens except for tokens later in the same train of thought. 
    '''
    debug_print("embeddings right before positional encodings", x, flag=self.debug)
    x = self.dual_positional_encoding(x, thoughts_taken)
    causal_mask = self.generate_thoughtsformer_mask(thoughts_taken).to(x.device)
    debug_print("embeddings right before transformer forward", x, flag=self.debug)
    if self.debug == True:
      plt.imshow(causal_mask)
      plt.show()
    temp_padding_mask = F.pad(torch.repeat_interleave(padding_mask, repeats=thoughts_taken+1, dim=-1), (0,self.max_context_length-self.max_sequence_length*(thoughts_taken+1)),value=True)
    x = self.transformer(x, mask=causal_mask, src_key_padding_mask=temp_padding_mask)
    debug_print("embeddings right after transformer forward", x, flag=self.debug)
    return x




class ThoughtsFormer(nn.Module):
  '''
  Model architecture that generates <code>max_thought_len</code> tokens before predicting the
  next token. 
  '''
  def __init__(self, 
               max_thought_len: int,
               vocab_size: int, 
               max_context_length: int = None,
               max_sequence_length: int = None, 
               reinforcement_learning: bool = False,
               sinusoidal_position_encoding: bool = False,
               num_layers: int = 8, 
               n_head: int = 8, 
               d_embed: int = 768, 
               dim_feedforward: int = 2048, 
               dropout: int = 0.1, 
               activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu, 
               layer_norm_eps: float = 0.0001, 
               batch_first: bool = True, 
               norm_first: bool = True, 
               bias: bool = True, 
               verbose: bool = False
              ):
    '''Initalization method of the ThoughtsFormer

    Args:
        max_thought_len (int): The maximum length of any train of thought. Set to 0 to function as a normal transformer
        vocab_size (int): vocab_size
        max_context_length (int, optional): The maximum context length of the model. Must specify this or specify max_sequence_length because max_sequence_length * (max_thought_len + 1) is equal to max_context_length. Defaults to None.
        max_sequence_length (int, optional): The maximum sequence length of the model. Must specify this or specify max_context_length because max_sequence_length * (max_thought_len + 1) is equal to max_context_length. Defaults to None.
        reinforcement_learning (bool, optional): Whether to initialize the output heads policy_feedforward and value_feedforward or the head out. Necessary for calling methods like forward_rl_tokens. Defaults to False.
        sinusoidal_position_encoding (bool, optional): Whether or not to use sinusoidal or learned positional encodings. Positional encodings are 2D, for position in thought and position in sequence; thought encodings will always be learned, this parameter specifies if position in sequence tokens are sinusoidal. Defaults to False.
        num_layers (int, optional): Unchanged transformer parameter. Defaults to 8.
        n_head (int, optional): Unchanged transformer parameter. Defaults to 8.
        d_embed (int, optional): Unchanged transformer parameter. Defaults to 768.
        dim_feedforward (int, optional): Unchanged transformer parameter. Defaults to 2048.
        dropout (int, optional): Unchanged transformer parameter. Defaults to 0.1.
        activation (Callable[[torch.Tensor], torch.Tensor], optional): Unchanged transformer parameter. Defaults to F.gelu.
        layer_norm_eps (float, optional): Unchanged transformer parameter. Defaults to 0.0001.
        batch_first (bool, optional): Unchanged transformer parameter. Defaults to True.
        norm_first (bool, optional): Unchanged transformer parameter. Defaults to True.
        bias (bool, optional): Unchanged transformer parameter. Defaults to True.
        verbose (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: Raises a ValueError if max_context_length and max_sequence_length are not specified, or if both are passed in with conflicting values
        RuntimeError: For unreachable code.
    '''

    super().__init__()
    
    if max_context_length is None and max_sequence_length is None:
      raise ValueError("Must specify either max_context_length or max_sequence_length")
    elif not max_context_length is None and not max_sequence_length is None and max_context_length != max_sequence_length * (max_thought_len + 1):
      raise ValueError(f"Error in maximum context length and maximum sequence length specifications. The maximum context length must be equal to max_sequence_length * (max_thought_length + 1), but in the input max_context_length is {max_context_length} and max_sequence_length * (max_thought_len + 1) is { max_sequence_length * (max_thought_len + 1)}")
    elif max_context_length is None: 
      max_context_length = max_sequence_length * (max_thought_len + 1) 
    
    
    if max_context_length is None:
      raise RuntimeError("Max context length undefined--unreachable code reached")

    if max_context_length % (max_thought_len+1) != 0:
      raise NotImplementedError("Number of thoughts + 1 currently must evenly divide into max_context_length")

    self.max_context_length, self.d_embed = max_context_length, d_embed
    self.max_thought_length = max_thought_len
    self.max_sequence_length = self.max_context_length // (self.max_thought_length + 1)
    self.vocab_size = vocab_size
    self.transformer = ThoughtCausalTransformer(
      max_thought_len=max_thought_len, 
      max_context_length=max_context_length, 
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
      debug=verbose
      )
    
    
    # self.policy_feedforward = nn.Sequential(
    #   nn.Linear(d_embed, d_embed),
    #   nn.GELU(),
    #   nn.Linear(d_embed, vocab_size)
    # )
    if reinforcement_learning:
      self.policy_feedforward = nn.Linear(d_embed, vocab_size,bias=False)
      # Weird difference between the two but GPT2 uses the single head for the feedforward head and I want to do the weight transfer entirely but allow extra modularity for the value function
      self.value_feedforward = nn.Sequential(
        nn.Linear(d_embed, dim_feedforward),
        nn.GELU(),
        nn.Linear(dim_feedforward, 1)
      )
    else:
      self.out = nn.Linear(d_embed, vocab_size)
      
    self.token_embedding = nn.Embedding(vocab_size, d_embed)
    self.tau = 0.5 # for gumbel softmax
    self.debug = verbose

  def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor, use_gumbel: bool = True) -> torch.Tensor:
    '''
    Takes in tokens, iteratively adds thoughts to those tokens by predicting the next thought token and sampling from the distribution, then returns the final context with thoughts added along with the logits for the final token predictions.
    '''
    assert tokens.size(1) <= self.max_sequence_length # Embeddings passed in are just the first sequence no padding
    
    
    state_embeddings = self.token_embedding(tokens)
    state_embeddings = self.prepare_thoughtsformer_embedding_input(state_embeddings)
    padding_mask = self._prepare_thoughtsformer_padding_mask_input(padding_mask)
    
    debug_tokens = torch.zeros_like(state_embeddings[:,:,0])
    debug_tokens[:, :self.max_sequence_length] = tokens
    
    from dual_positional_encoding import simple_batched_reshape_with_offset, inverse_simple_batched_reshape_with_offset
    for thought_iter in range(self.max_thought_length + 1):
      selected_next_embeddings = self.get_tokens_at_action_location(
            self.forward_embeddings(state_embeddings, padding_mask, thought_iter), thought_iter
      )
      if thought_iter != self.max_thought_length:
        if use_gumbel:
          selected_next_embeddings, selected_tokens = self.gumbel_calculation(selected_next_embeddings)
          # debug steps
          debug_tokens = token_batched_reshape_with_offset(debug_tokens, self.max_sequence_length, thought_iter)
          debug_tokens[:,:,thought_iter+1] = selected_tokens
          debug_tokens = token_inverse_simple_batched_reshape_with_offset(debug_tokens, thought_iter+1)
        state_embeddings = simple_batched_reshape_with_offset(state_embeddings, self.max_sequence_length, thought_iter)
        state_embeddings[:,:,thought_iter+1, :] = selected_next_embeddings
        # plt.imshow(self.state[0,:10]); plt.show()
        state_embeddings = inverse_simple_batched_reshape_with_offset(state_embeddings, thought_iter+1)
        
        
     
    logits = self.out(selected_next_embeddings)

    return logits
  
  def gumbel_calculation(self, selected_next_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    next_token_logits = self.out(selected_next_embeddings)
    if self.training:
      self.tau = max(self.tau * 0.999, 0.1)
      selected = F.gumbel_softmax(next_token_logits, tau=self.tau, hard=True)
      selected_tokens = selected.argmax(dim=-1)
      return selected @ self.token_embedding.weight, selected_tokens
    else:
      index = next_token_logits.argmax(dim=-1)
      return self.token_embedding(index), index
  
   

  def forward_embeddings(self, state_embeddings: torch.Tensor, padding_mask: torch.Tensor, n_thoughts_taken: int | torch.Tensor) -> torch.Tensor:
    if type(n_thoughts_taken) == torch.Tensor and n_thoughts_taken.numel() == 1:
      n_thoughts_taken: int  = n_thoughts_taken.item()
       
    assert n_thoughts_taken <= self.max_thought_length, f"n_thoughts_taken ({n_thoughts_taken}) > thought_len ({self.max_thought_length})"
    assert state_embeddings.ndim == 3, f"state_embeddings must be three dimensional, where the dimensons are (batch_size, seq_len, ndim). Instead recieved {state_embeddings.ndim} dimensional input"
    assert state_embeddings.size(2) == self.d_embed, f"state_embedding's final dimension must be of size d_embed ({self.d_embed}), instead recieved size ({state_embeddings.size(2)})"
    
    assert state_embeddings.size(0) == padding_mask.size(0), f"mismatch between the size of dimension 0 in state_embedding ({state_embeddings.size(0)}) and the size of dimension 0 in padding_mask ({padding_mask.size(0)}). This is the batch size so these dimensions should be equal values"
    
    state_embeddings = self.prepare_thoughtsformer_embedding_input(state_embeddings)
    
    padding_mask = self._prepare_thoughtsformer_padding_mask_input(padding_mask)

    # print(torch.sum(padding_mask == 0,dim=1))
    # print( int(torch.sum(padding_mask == 0,dim=1).max()))

    next_embeddings = self.transformer(state_embeddings, padding_mask, n_thoughts_taken)
    debug_print("future embeddings\n", next_embeddings,  flag=self.debug)
    
    return next_embeddings
  
  def forward_rl_embeddings(self, state_embeddings: torch.Tensor, padding_mask: torch.Tensor, n_thoughts_taken: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Takes in the current state, the padding mask, and the number of thoughts already taken. Outputs the logits for the next action and value for every token in the sequence. 
    '''
    next_embeddings = self.forward_embeddings(state_embeddings, padding_mask, n_thoughts_taken)
    action_embeddings = self.get_tokens_at_action_location(next_embeddings, n_thoughts_taken)
    
    return self.policy_feedforward(action_embeddings), self.value_feedforward(action_embeddings).squeeze(dim=2)
    
  # Claude's version for tokens!
  def forward_rl_tokens(self, tokens: torch.Tensor, padding_mask: torch.Tensor, n_thoughts_taken: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Accepts raw tokens, embeds them, and then calls the existing forward_rl_embeddings method. 
    Will complete one step of the thoughtsformer process depending on n_thoughts_taken. Ideally this is 
    called iteratively until n_thoughts_taken is equal to max_thought_len, in which case 
    the output logits can be used for next token prediction 
    
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
    
    # Call the existing forward_rl_embeddings method
    return self.forward_rl_embeddings(state_embeddings, padding_mask, n_thoughts_taken)
  
  def entire_thought_generation(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Takes in tokens, iteratively adds thoughts to those tokens by predicting the next thought token and sampling from the distribution, then returns the final context with thoughts added along with the logits for the final token predictions.
    '''
    tokens = F.pad(tokens, (0,self.max_context_length-tokens.size(1)))

    for thought_iter in range(self.max_thought_length + 1):
      logits, _ = self.forward_rl_tokens(tokens, padding_mask, thought_iter)
      sampled_tokens, _ = self._sample_tokens(logits)
      if thought_iter != self.max_thought_length:
        tokens = token_batched_reshape_with_offset(tokens.view(1,-1), self.max_sequence_length, thought_iter)
        tokens[:,:,thought_iter+1] = sampled_tokens
        # plt.imshow(self.state[0,:10]); plt.show()
        tokens = tokens.view(1,-1)
    final_step_logits = logits
    return tokens, final_step_logits
   
   
   
  def _sample_tokens(self, logits: torch.Tensor):
    '''
    Samples tokens from logits.
    '''
    log_probs = F.log_softmax(logits, dim=-1)
    
    probs = F.softmax(logits, dim = -1)
    sampled_tokens = torch.multinomial(
        probs.view(-1, probs.size(-1)),
        num_samples=1
    ).view(-1, probs.size(1))
    
    action_probs = log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    
    return sampled_tokens, action_probs
       
  def prepare_thoughtsformer_embedding_input(self, x: torch.Tensor):
    '''
    Internal method. Pads embeddings to full context length.
    '''
    # Expected shape batch_size, seq_len, d_embed
    # Want to zero pad the seq_len dimension to be max_seq_len + max_seq_len * thought_length
    max_context_length = self.max_context_length
    
    seq_len = x.size(1) 
    assert seq_len <= max_context_length, f"Length of the input embeddings ({seq_len}) exceeds maximum context length ({max_context_length}). Sequence length is dimension 1 of the input embeddings."
    
    return F.pad(x, (0,0,0,max_context_length - seq_len))

  def _prepare_thoughtsformer_padding_mask_input(self, padding_mask: torch.Tensor) -> torch.Tensor:
    '''
    Internal method. Pads the padding mask to full context length.
    '''
    # Expected shape batch_size, seq_len
    # Want to zero pad the seq_len dimension to be max_seq_len + max_seq_len * thought_length
    max_sequence_length = self.max_sequence_length
  
    seq_len = padding_mask.size(1) 
    assert seq_len <= max_sequence_length, f"Length of the padding mask's ({seq_len}) exceeds maximum sequence length ({max_sequence_length}). Sequence length is dimension 1 of the input embeddings."
    
    return F.pad(padding_mask, (0,max_sequence_length - seq_len), value=1)
      
  def get_tokens_at_action_location(self, embeddings_or_logits: torch.Tensor, thought_length: int) -> torch.Tensor:
    n_real_tokens = self.max_context_length // (self.max_thought_length + 1)
    if isinstance(thought_length, torch.Tensor):
      token_predictor_locations = (torch.arange(n_real_tokens).to(thought_length.device) + 1) * (thought_length + 1) - 1
    else:
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
  def from_pretrained_GPT2(cls: Type['ThoughtsFormer'], n_thoughts: int = 0, reinforcement_learning: bool = False) -> 'ThoughtsFormer':
    from transformers import GPT2LMHeadModel
    pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
    
    max_sequence_length = 1024//(n_thoughts+1)
    thoughtsformer = cls(
      max_thought_len=n_thoughts,
      vocab_size=50257,
      max_context_length=1024,
      reinforcement_learning=reinforcement_learning,
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
        'transformer.transformer.norm.weight' : 'transformer.ln_f.weight',
        'transformer.transformer.norm.bias' : 'transformer.ln_f.bias',
        'transformer.dual_positional_encoding.learned_positional_encoding.weight' : 'transformer.wpe.weight'
    }

    if reinforcement_learning:
      explicit_map['policy_feedforward.weight'] = 'lm_head.weight'
    else:
      explicit_map['out.weight'] = 'lm_head.weight'

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
    
    
def token_batched_reshape_with_offset(x: torch.Tensor, max_seq_length: int, thoughts_taken: int) -> torch.Tensor:
  thoughts = thoughts_taken + 1
  max_thoughts = x.size(1) // max_seq_length
  x = x[:,:max_seq_length*thoughts].view(x.size(0), max_seq_length, thoughts)
  return F.pad(x,(0, (max_thoughts - thoughts)))

def token_inverse_simple_batched_reshape_with_offset(x: torch.Tensor, thoughts_taken: int) -> torch.Tensor:
  seq_len, max_thoughts = x.size(1), x.size(2)
  x = x[:,:,:thoughts_taken+1].reshape(x.size(0), -1)
  x = F.pad(x,(0, seq_len * (max_thoughts - (thoughts_taken+1))))
  return x