import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium import Env, spaces
from tiny_shakespeare import TinyShakespeareDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

def token_batched_reshape_with_offset(x: torch.Tensor, max_seq_length: int, thoughts_taken: int) -> torch.Tensor:
    thoughts = thoughts_taken + 1
    max_thoughts = x.size(1) // max_seq_length
    x = x[:,:max_seq_length*thoughts].view(x.size(0), max_seq_length, thoughts)
    return F.pad(x,(0, (max_thoughts - thoughts)))
        
class ThoughtsFormerEnv(Env):
    def __init__(self, vocab_size, max_sequence_length, max_thought_length):
        super(ThoughtsFormerEnv, self).__init__()
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length 
        self.max_thought_length = max_thought_length
        self.max_context_length = max_sequence_length * (max_thought_length+1)
        
        # Logits
        self.action_space = spaces.Box(low=-100, high=100, shape=(max_sequence_length,vocab_size), dtype=np.float32)

        
        self.observation_space = spaces.Dict({
            "state" : spaces.MultiDiscrete([vocab_size] * self.max_context_length),
            "thought_step" : spaces.Discrete(max_thought_length+1)
        })
        
        self.dataset = TinyShakespeareDataset(max_sequence_length,window_offset=max_sequence_length//4)
        self.dataset_len = len(self.dataset)
        self.dataset_iter = 0
        
        self.thought_step = 0

    def reset(self, seed=None):
        super().reset(seed=seed)  # Ensures Gymnasium's seeding is properly handled
        
        self.state, self.labels = self.dataset[self.dataset_iter]
        self.state = F.pad(self.state, (0,self.max_context_length-self.max_sequence_length))
     
        # prepare self.state and massively elongate
        self.dataset_iter += 1
        if self.dataset == self.dataset_len:
            self.dataset_iter = 0

        self.thought_step = 0
        
        obs = {
            'state' : self.state.numpy(),
            'thought_step' : self.thought_step
        }
        return obs, {}

    def step(self, action):
        self.state = self.state.view(1,-1)
        
        probs = F.softmax(torch.from_numpy(action), dim = -1)
        sampled_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(-1, probs.size(0))
            
        
        if self.thought_step == self.max_thought_length:
            reward = self.reward(action)
            # print(reward.shape)
            done = True
        else:
            reward = torch.zeros(self.max_sequence_length)
            done = False
            # Add the thought!
            self.state = token_batched_reshape_with_offset(self.state, self.max_sequence_length, self.thought_step) # (batch x max_seq_len, (max_thought_len+1))
            # before
            # plt.imshow(self.state[0,:10]); plt.show()
            
            self.state[:,:,self.thought_step+1] = sampled_tokens
            # plt.imshow(self.state[0,:10]); plt.show()
            self.state =  self.state.view(1,-1)
        
        self.state = self.state.view(-1)
        obs = {
            'state' : self.state.numpy(),
            'thought_step' : self.thought_step
        }
        
        info = {'reward' : reward, 'actions_taken' : sampled_tokens.squeeze(0)} 
        
        # print(info)
        self.thought_step += 1
        return obs, -np.inf, done, False, info #obs, reward, done, truncated, info

    def reward(self, action):
        return -F.cross_entropy(torch.tensor(action), self.labels, reduction='none')
    
# Check the environment to ensure compatibility
from token_level_ppo import TokenLevelPPO, TokenLevelRolloutBuffer, ThoughtsFormerPolicy
env = ThoughtsFormerEnv(vocab_size=50257, max_sequence_length=512,max_thought_length=1)

ppo = TokenLevelPPO(ThoughtsFormerPolicy, env, n_steps=4, batch_size=2, max_sequence_length=512, verbose=2)
for i in range(3):
    ppo.learn(4, log_interval=1)