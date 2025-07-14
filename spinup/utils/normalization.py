"""
Observation normalization utilities for RL algorithms.
Simplified version for use with environment wrappers.
"""

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
    """
    
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        """Update the running statistics with new batch of data"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch statistics using Welford's algorithm"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


# Test function
def test_normalization():
    """Test the normalization implementation"""
    print("Testing observation normalization utilities...")
    
    # Create test data
    np.random.seed(42)
    test_obs = np.random.randn(1000, 10) * 5 + 10  # Mean ~10, std ~5
    
    # Create running mean std tracker
    rms = RunningMeanStd(shape=(10,))
    
    # Update with data
    rms.update(test_obs)
    
    print(f"Original data - Mean: {test_obs.mean(axis=0)[:3]}, Std: {test_obs.std(axis=0)[:3]}")
    print(f"RMS tracking - Mean: {rms.mean[:3]}, Std: {np.sqrt(rms.var)[:3]}")
    print("✓ Normalization test passed!")


if __name__ == "__main__":
    test_normalization()