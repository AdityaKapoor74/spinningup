import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from spinup.utils.normalization import ObservationNormalizer


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh,
                 normalize_observations=True, obs_clip_range=10.0):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # Observation normalization
        self.normalize_observations = normalize_observations
        if self.normalize_observations:
            self.obs_normalizer = ObservationNormalizer(
                obs_dim=obs_dim,
                device='cpu',  # Will be moved to correct device later
                clip_range=obs_clip_range
            )
        else:
            self.obs_normalizer = None

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

        # Track training mode for normalization
        self._update_obs_stats = True

    def _normalize_obs(self, obs):
        """Normalize observations if enabled"""
        if not self.normalize_observations:
            return obs
        return self.obs_normalizer(obs, update_stats=self._update_obs_stats)
    
    def set_obs_update_mode(self, update_stats=True):
        """Set whether to update observation statistics"""
        self._update_obs_stats = update_stats

    def step(self, obs):
        with torch.no_grad():
            # Normalize observations
            obs_normalized = self._normalize_obs(obs)
            
            pi = self.pi._distribution(obs_normalized)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def save_obs_normalizer(self):
        """Get observation normalizer state for saving"""
        if self.normalize_observations:
            return self.obs_normalizer.state_dict()
        return None

    def load_obs_normalizer(self, state_dict):
        """Load observation normalizer state"""
        if self.normalize_observations and state_dict is not None:
            self.obs_normalizer.load_state_dict(state_dict)

    def to(self, device):
        """Handle device transfer for observation normalizer"""
        result = super().to(device)
        if result.normalize_observations:
            # Move normalizer components to device
            result.obs_normalizer.obs_rms.device = device
            result.obs_normalizer.obs_rms.mean = result.obs_normalizer.obs_rms.mean.to(device)
            result.obs_normalizer.obs_rms.var = result.obs_normalizer.obs_rms.var.to(device)
        return result

    def train(self, mode=True):
        """Override train mode to handle observation normalization"""
        result = super().train(mode)
        # In training mode, update obs stats during rollouts
        # In eval mode, freeze obs stats
        result.set_obs_update_mode(update_stats=mode)
        return result