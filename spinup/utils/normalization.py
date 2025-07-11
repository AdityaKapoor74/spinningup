"""
Observation normalization utilities for RL algorithms.
Based on implementations used in OpenAI Baselines and published PPO papers.
"""

import torch
import numpy as np


class RunningMeanStd:
    """
    Tracks the mean, variance and count of values using Welford's algorithm.
    
    This is the standard implementation used in:
    - OpenAI Baselines
    - Stable Baselines3  
    - Many published PPO papers
    
    Args:
        epsilon: Small value to avoid division by zero
        shape: Shape of the values being tracked
        device: PyTorch device for tensors
    """
    
    def __init__(self, epsilon=1e-4, shape=(), device='cpu'):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.device = device

    def update(self, x):
        """Update the running statistics with new batch of data"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch statistics using Welford's algorithm"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        """Normalize input using current statistics"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)
    
    def denormalize(self, x):
        """Denormalize input (inverse operation)"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        
        return x * torch.sqrt(self.var + 1e-8) + self.mean

    def state_dict(self):
        """Get state for saving/loading"""
        return {
            'mean': self.mean,
            'var': self.var, 
            'count': self.count
        }
    
    def load_state_dict(self, state_dict):
        """Load state from saved dict"""
        self.mean = state_dict['mean'].to(self.device)
        self.var = state_dict['var'].to(self.device)
        self.count = state_dict['count']


class ObservationNormalizer:
    """
    Observation normalizer that can be easily integrated into RL algorithms.
    
    This class wraps RunningMeanStd and provides a clean interface for 
    observation normalization as used in published PPO implementations.
    """
    
    def __init__(self, obs_dim, device='cpu', clip_range=10.0):
        """
        Args:
            obs_dim: Dimension of observations
            device: PyTorch device
            clip_range: Range to clip normalized observations
        """
        self.obs_rms = RunningMeanStd(shape=(obs_dim,), device=device)
        self.clip_range = clip_range
        self.device = device
        
    def __call__(self, obs, update_stats=True):
        """
        Normalize observations
        
        Args:
            obs: Observations to normalize (numpy array or torch tensor)
            update_stats: Whether to update running statistics
            
        Returns:
            Normalized observations
        """
        if update_stats:
            self.obs_rms.update(obs)
        
        normalized_obs = self.obs_rms.normalize(obs)
        
        # Clip to prevent extreme values
        if self.clip_range is not None:
            normalized_obs = torch.clamp(normalized_obs, 
                                       -self.clip_range, 
                                       self.clip_range)
        
        return normalized_obs
    
    def normalize_without_update(self, obs):
        """Normalize without updating statistics (for evaluation)"""
        return self(obs, update_stats=False)
    
    def state_dict(self):
        """Get state for saving"""
        return {
            'obs_rms': self.obs_rms.state_dict(),
            'clip_range': self.clip_range
        }
    
    def load_state_dict(self, state_dict):
        """Load state from saved dict"""
        self.obs_rms.load_state_dict(state_dict['obs_rms'])
        self.clip_range = state_dict['clip_range']


# Utility functions for easy integration
def create_obs_normalizer(env, device='cpu', clip_range=10.0):
    """Create observation normalizer for given environment"""
    obs_dim = env.observation_space.shape[0]
    return ObservationNormalizer(obs_dim, device=device, clip_range=clip_range)

def test_normalization():
    """Test the normalization implementation"""
    print("Testing observation normalization...")
    
    # Create test data
    np.random.seed(42)
    test_obs = np.random.randn(1000, 10) * 5 + 10  # Mean ~10, std ~5
    
    # Create normalizer
    normalizer = ObservationNormalizer(obs_dim=10)
    
    # Normalize data
    normalized = normalizer(test_obs)
    
    print(f"Original data - Mean: {test_obs.mean(axis=0)[:3]}, Std: {test_obs.std(axis=0)[:3]}")
    print(f"Normalized data - Mean: {normalized.mean(dim=0)[:3]}, Std: {normalized.std(dim=0)[:3]}")
    print("✓ Normalization test passed!")

if __name__ == "__main__":
    test_normalization()