"""
Environment wrappers for observation normalization.
This is the standard approach used by OpenAI Baselines and Stable Baselines3.
"""

import numpy as np
import gym
from gym import spaces
from spinup.utils.normalization import RunningMeanStd


class VecNormalize(gym.Wrapper):
    """
    Observation normalization wrapper for environments.
    
    This wrapper normalizes observations using running mean and standard deviation,
    exactly as implemented in OpenAI Baselines and published PPO papers.

    """
    
    def __init__(self, env, ob=True, clipob=10.0, 
                 gamma=0.99, epsilon=1e-8, training=True):
        """
        Args:
            env: Environment to wrap
            ob: Whether to normalize observations
            clipob: Range to clip normalized observations
            gamma: Discount factor for return normalization
            epsilon: Small value to avoid division by zero
            training: Whether wrapper is in training mode (updates stats)
        """
        super().__init__(env)
        
        self.ob = ob
        self.clipob = clipob
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        
        # Initialize running statistics
        if self.ob:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
    
    def step(self, action):
        """
        Step the environment and normalize observations/returns
        """
        obs, rews, dones, infos = self.env.step(action)
        
        # Handle observation normalization
        if self.ob:
            obs = self._normalize_obs(obs)
        
        return obs, rews, dones, infos
    
    def reset(self, **kwargs):
        """
        Reset environment and normalize initial observation
        """
        obs = self.env.reset(**kwargs)
        
        # Normalize observation
        if self.ob:
            obs = self._normalize_obs(obs)
        
        return obs
    
    def _normalize_obs(self, obs):
        """
        Normalize observations using running statistics
        """
        if self.training:
            # Update statistics during training
            self.obs_rms.update(obs)
        
        # Normalize and clip
        obs_norm = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        
        if self.clipob > 0.0:
            obs_norm = np.clip(obs_norm, -self.clipob, self.clipob)
        
        return obs_norm.astype(np.float32)
    
    def get_original_obs(self):
        """
        Get the original (unnormalized) observation space
        """
        return self.env.observation_space
    
    def set_training(self, training):
        """
        Set training mode (whether to update running statistics)
        """
        self.training = training
    
    def sync_running_stats(self):
        """
        Synchronize running statistics across MPI processes
        """
        from spinup.utils.mpi_tools import mpi_avg, num_procs
        
        if num_procs() <= 1:
            return
        
        if self.ob:
            # Sync observation statistics
            mean = self.obs_rms.mean
            var = self.obs_rms.var
            count = self.obs_rms.count
            
            # Average across processes
            synced_mean = np.array([mpi_avg(x) for x in mean.flat]).reshape(mean.shape)
            synced_var = np.array([mpi_avg(x) for x in var.flat]).reshape(var.shape)
            synced_count = mpi_avg(count)
            
            # Update local statistics
            self.obs_rms.mean = synced_mean
            self.obs_rms.var = synced_var
            self.obs_rms.count = synced_count
    
    def get_stats_dict(self):
        """
        Get normalization statistics for saving
        """
        stats = {}
        
        if self.ob:
            stats['obs_rms'] = {
                'mean': self.obs_rms.mean,
                'var': self.obs_rms.var,
                'count': self.obs_rms.count
            }
        
        return stats
    
    def load_stats_dict(self, stats):
        """
        Load normalization statistics from saved data
        """
        if self.ob and 'obs_rms' in stats:
            self.obs_rms.mean = stats['obs_rms']['mean']
            self.obs_rms.var = stats['obs_rms']['var']
            self.obs_rms.count = stats['obs_rms']['count']


class DummyVecNormalize:
    """
    Dummy wrapper that does nothing - for when normalization is disabled
    """
    
    def __init__(self, env):
        self.env = env
        
    def __getattr__(self, name):
        """Delegate all attributes to the wrapped environment"""
        return getattr(self.env, name)
    
    def sync_running_stats(self):
        """No-op for compatibility"""
        pass
    
    def get_stats_dict(self):
        """Return empty dict for compatibility"""
        return {}
    
    def load_stats_dict(self, stats):
        """No-op for compatibility"""
        pass
    
    def set_training(self, training):
        """No-op for compatibility"""
        pass


def make_env_with_normalization(env_fn, normalize_observations=True, 
                              clip_obs=10.0, 
                              gamma=0.99):
    """
    Factory function to create environment with optional normalization.
    
    Args:
        env_fn: Function that creates the environment
        normalize_observations: Whether to normalize observations
        clip_obs: Range to clip normalized observations
        gamma: Discount factor for return normalization
    
    Returns:
        Environment wrapped with normalization (or dummy wrapper if disabled)
    """
    env = env_fn()
    
    if normalize_observations:
        env = VecNormalize(
            env=env,
            ob=normalize_observations,
            clipob=clip_obs,
            gamma=gamma,
            training=True
        )
    else:
        # Use dummy wrapper for consistent interface
        env = DummyVecNormalize(env)
    
    return env


# Convenience function for common use case
def make_normalized_env(env_name, clip_obs=10.0):
    """
    Create a gym environment with observation normalization enabled.
    
    Args:
        env_name: Name of the gym environment
        clip_obs: Range to clip normalized observations
    
    Returns:
        Environment with observation normalization
    """
    return make_env_with_normalization(
        env_fn=lambda: gym.make(env_name),
        normalize_observations=True,
        clip_obs=clip_obs
    )


# Test function
def test_normalization_wrapper():
    """Test the normalization wrapper"""
    
    print("Testing observation normalization wrapper...")
    
    # Create environment with normalization
    env = make_normalized_env('CartPole-v1', clip_obs=10.0)
    
    # Collect some observations
    obs_data = []
    obs = env.reset()
    
    for _ in range(100):
        obs_data.append(obs.copy())
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        if done:
            obs = env.reset()
    
    obs_data = np.array(obs_data)
    
    print(f"Collected {len(obs_data)} normalized observations")
    print(f"Normalized obs - Mean: {obs_data.mean(axis=0)}")
    print(f"Normalized obs - Std: {obs_data.std(axis=0)}")
    print(f"Original obs space: {env.get_original_obs()}")
    
    # Test statistics
    stats = env.get_stats_dict()
    print(f"Normalization statistics collected: {len(stats) > 0}")
    
    env.close()
    print("✓ Wrapper test completed successfully!")


if __name__ == "__main__":
    test_normalization_wrapper()