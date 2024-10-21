
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import DictRolloutBuffer
import torch.nn.functional as F
import numpy as np
from gymnasium import Env, spaces

import torch
import torch.nn as nn
from thoughtsformer import ThoughtsFormer, simple_batched_reshape_with_offset
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.utils import obs_as_tensor
from torch.distributions import Categorical

class ThoughtsFormerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def _build(self, lr_schedule):
        self.model = ThoughtsFormer.from_pretrained_GPT2(1)
        # Override this method to prevent building the default MLP
        # Instead, set up your custom model components here
        
        # Example: Set up your action distribution
        self.action_dist = CategoricalDistribution(self.action_space)
        
        # Set up your optimizer
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    
    def forward(self, obs: dict):
        assert 'state' in obs and 'thought_step' in obs, f"'state' and 'thought_step' should be keys of the observation dictionary"
        
        logits, values = self.model.forward_ppo_with_tokens(obs['state'], torch.zeros_like(obs['state']), obs['thought_step'])
        return logits, values

        
    def evaluate_actions(self, obs: dict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits,dim=-1)
        
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        entropy = -(probs * log_probs).sum(-1) # 
        return values, action_log_probs, entropy

    def _get_action_dist_from_obs(self, obs):
        return self.model.forward_ppo_with_tokens(obs['state'], torch.zeros_like(obs['state']), obs['thought_step'])[0]

    def predict_values(self, obs):
        return self.model.forward_ppo_with_tokens(obs['state'], torch.zeros_like(obs['state']), obs['thought_step'])[1]




class TokenLevelRolloutBuffer(DictRolloutBuffer):
    def __init__(self, *args, max_sequence_length: int, **kwargs) -> None:
        self.max_sequence_length = max_sequence_length
        super().__init__(*args, **kwargs)
        
    def reset(self) -> None:
        super().reset()
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.max_sequence_length), dtype=np.float32)
        self.values = np.zeros_like(self.rewards)
        self.actions = np.zeros_like(self.rewards) 
        self.advantages = np.zeros_like(self.rewards)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs, self.action_dim))
    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.max_sequence_length)) # CHANGE IN EMMETT'S CODE (kinda janky change)
        log_prob = log_prob.reshape((self.n_envs, self.action_dim)) # CHANGE IN EMMETT'S CODE 
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = np.array(log_prob) # CHANGE IN EMMETT'S CODE 
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: torch.Tensor | np.ndarray, dones: np.ndarray) -> None:
        """
        Adapted for token-level GAE
        """
        if isinstance(last_values, torch.Tensor):
            last_values = last_values.cpu().numpy() 
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # Reshape advantages and returns to match token-level structure
        self.advantages = self.advantages.reshape(self.buffer_size, -1, self.max_sequence_length)
        self.returns = self.advantages + self.values

class TokenLevelPPO(PPO):
    def __init__(self, policy, env, max_sequence_length, **kwargs):
        self.max_sequence_length = max_sequence_length
        super().__init__(policy, env, **kwargs)
        self.rollout_buffer = TokenLevelRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            max_sequence_length=max_sequence_length,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
        )
        
    
    @classmethod
    def load(  # noqa: C901
        cls,
        path,
        max_sequence_length,
        env = None,
        device = "auto",
        custom_objects = None,
        print_system_info = False,
        force_reset = True,
        **kwargs,
    ):
        from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
        from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
        from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env
        from stable_baselines3.common.utils import (
            check_for_correct_spaces,
            get_device,
            get_schedule_fn,
            get_system_info,
            set_random_seed,
            update_learning_rate,
        )
        import warnings
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            max_sequence_length=max_sequence_length,
            _init_setup_model=False,  # type: ignore[call-arg]
            **kwargs
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
            
            
        
        return model
    
    def _setup_model(self) -> None:
        super()._setup_model()
        self.rollout_buffer = TokenLevelRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            max_sequence_length=self.max_sequence_length,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
        )
        

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        # Reset or initialize variables
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        
        rollout_buffer.reset()
        for step in range(n_rollout_steps):
            # Get actions and values from the policy
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                log_probs, values = self.policy(obs_tensor)
            log_probs = log_probs.cpu().numpy()

            # Rescale and perform action
            clipped_actions = log_probs

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(log_probs, self.action_space.low, self.action_space.high)

            new_observations, rewards, dones, infos = env.step(clipped_actions)
            rewards = torch.stack([infos[idx]['reward'] for idx in range(len(infos))]).numpy()
            actions = torch.stack([infos[idx]['actions_taken'] for idx in range(len(infos))]).numpy()
            
            self.num_timesteps += env.num_envs
            
            self._update_info_buffer(infos, dones)
            
            
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

  
            
            # Store data in the buffer
            
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs)
            
            self._last_obs = new_observations  
            self._last_episode_starts = dones
            
        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_observations, self.device))  # type: ignore[arg-type]
            
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        return True
        
    def train(self) -> None:
        counter = 0
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                
                
                actions = rollout_data.actions
                actions = actions.reshape(actions.size(0), self.max_sequence_length).to(torch.long)
                # actions = actions.reshape(actions.size(0),self.
                rollout_data.observations['state'] = rollout_data.observations['state'].to(torch.long)
                rollout_data.observations['thought_step'] = int(rollout_data.observations['thought_step'][0].item())
                values, action_log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                values = values.view(-1, actions.shape[1]) # (batch x tokens)
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = advantages.view(-1, actions.shape[1]) # (batch x tokens)
                advantages = (advantages - advantages.mean(dim=0, keepdim=True)) / (advantages.std(dim=0, keepdim=True) + 1e-8)
                
                # ratio between old and new policy, should be one at the first iteration
               
                action_log_prob = action_log_prob.view(-1, actions.shape[1]) # what our policy thinks about the old action taken
                old_log_prob = rollout_data.old_log_prob.view(action_log_prob.size(0), actions.shape[1],-1) # what the old policy thinks about the action taken
                old_log_prob = F.log_softmax(old_log_prob, dim=-1)
                old_log_prob = old_log_prob.gather(-1, actions.unsqueeze(-1)).squeeze(-1) # FIX: Terrible extraction of old log prob for taken actions after the fact and requires storing an outrageous amount of information
                ratio = torch.exp(action_log_prob - old_log_prob)
                
                current_clip_range = self.clip_range(counter / self._total_timesteps)
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - current_clip_range, 1 + current_clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2)

                
                # Value clipping is omitted in the token-based version, but could be added if needed
                value_loss = F.mse_loss(rollout_data.returns.view(-1, actions.shape[1]), values, reduction='none')
                # Value loss using the TD(gae_lambda) target

                # Entropy loss favor exploration
                if entropy is None:
                    entropy_loss = -action_log_prob
                else:
                    entropy_loss = -entropy.view(-1, actions.shape[1])

                token_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                loss = token_loss.mean()
                
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
               
                # Logging
                counter+=1
            
                self.logger.record("train/loss", loss.item())
                self.logger.record("train/policy_loss", policy_loss.abs().mean().item())
                self.logger.record("train/value_loss", value_loss.mean().item())
                self.logger.record("train/entropy_loss", entropy_loss.mean().item())
                self.logger.record("train/mean_reward", rollout_data.returns.mean().item())
                
        self.logger.dump(step=counter)