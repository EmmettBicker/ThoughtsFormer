import torch
import torch.nn.functional as F
from thoughtsformer import ThoughtsFormer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution

class ThoughtsFormerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        
        self.max_thought_len = kwargs.pop("max_thought_len", None)
        self.from_pretrained = kwargs.pop("from_gpt2", None)
        self.has_trained_before = True
        self.model = kwargs.pop("thoughtsformer_model", None)
        if self.max_thought_len is None and self.from_pretrained == True:
            raise ValueError("Please specify max_thought_len in **kwargs. If you are calling from an algorithm definition such as PPO(ThoughtsFormerPolicy, ...), make sure to pass in a dictionary with all kwargs named policy_kwargs into that method.")
        if self.from_pretrained is None:
            raise ValueError("Please specify from_pretrained in **kwargs. If you are calling from an algorithm definition such as PPO(ThoughtsFormerPolicy, ...), make sure to pass in a dictionary with all kwargs named policy_kwargs into that method.")
        if self.model is None and self.from_pretrained == False:
            raise ValueError("If from_pretrained == False, must pass in {\"thoughtsformer_model\" : ThoughtsFormer(...)} in kwargs")
        
        if self.from_pretrained == False and not isinstance(self.model, ThoughtsFormer):
            raise ValueError("thoughtsformer_model must be of instance ThoughtsFormer")
        super().__init__(*args, **kwargs)
        

    def _build(self, lr_schedule):
        if self.from_pretrained:
            self.model = ThoughtsFormer.from_pretrained_GPT2(self.max_thought_len)
        else:
            pass
            # self.model = self.model already called
        # Override this method to prevent building the default MLP
        # Instead, set up your custom model components here
        
        # Example: Set up your action distribution
        self.action_dist = CategoricalDistribution(self.action_space)
        
        # Set up your optimizer
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        
    

    
    def forward(self, obs: dict):
        assert 'state' in obs and 'thought_step' in obs, f"'state' and 'thought_step' should be keys of the observation dictionary"
        
        logits, values = self.model.forward_rl_tokens(obs['state'], torch.zeros_like(obs['state']), obs['thought_step'])
        sampled_tokens, action_probs = self.sample_tokens(logits)
       
        return sampled_tokens, values, action_probs
    
    
    def forward_with_logits(self, obs: dict):
        assert 'state' in obs and 'thought_step' in obs, f"'state' and 'thought_step' should be keys of the observation dictionary"
        
        logits, values = self.model.forward_rl_tokens(obs['state'], torch.zeros_like(obs['state']), obs['thought_step'])
        sampled_tokens, action_probs = self.sample_tokens(logits)
       
        return sampled_tokens, values, action_probs, logits

    def sample_tokens(self, logits: torch.Tensor):
        log_probs = F.log_softmax(logits, dim=-1)
        
        probs = F.softmax(logits, dim = -1)
        sampled_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(-1, probs.size(1))
        
        action_probs = log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        
        return sampled_tokens, action_probs
                     
        
    def evaluate_actions(self, obs: dict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sorting! Or really grouping by timestep 
        
        thought_steps = obs["thought_step"].flatten()
        batch_size = thought_steps.numel()
        max_sequence_length = self.model.max_context_length // (self.model.max_thought_length + 1)
        action_log_probs = torch.zeros(batch_size, max_sequence_length).to(thought_steps.device)
        values = torch.zeros(batch_size, max_sequence_length).to(thought_steps.device)
        logits = torch.zeros(batch_size, max_sequence_length, self.model.vocab_size).to(thought_steps.device)
        
        for value in torch.unique(thought_steps):
            idxs = torch.where(thought_steps == value.item())[0]
            temp_obs = {
                'state' : obs['state'][idxs],
                'thought_step' : int(value.item())
            }
            
            _, values[idxs], _, logits[idxs] = self.forward_with_logits(temp_obs)
        log_probs = F.log_softmax(logits,dim=-1)
        probs = F.softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(-1) # 
        return values, action_log_probs, entropy



    def _get_action_dist_from_obs(self, obs):
        return self.model.forward_rl_tokens(obs['state'], torch.zeros_like(obs['state']), obs['thought_step'])[0]

    def predict_values(self, obs):
        return self.model.forward_rl_tokens(obs['state'], torch.zeros_like(obs['state']), obs['thought_step'])[1]

