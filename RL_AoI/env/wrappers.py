#!/usr/bin/env python3
"""
Environment Wrappers for RL Training.

Includes wrappers for:
- Converting discrete actions to continuous (for SAC)
- Action masking
- Observation normalization
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any


class DiscreteToBoxWrapper(gym.ActionWrapper):
    """
    Wrapper that converts a Discrete action space to a Box action space.
    
    This allows algorithms that require continuous actions (like SAC)
    to work with discrete action environments.
    
    The continuous action is mapped to the nearest discrete action.
    """
    
    def __init__(self, env: gym.Env):
        """
        Initialize the wrapper.
        
        Args:
            env: Environment with Discrete action space
        """
        super().__init__(env)
        
        assert isinstance(env.action_space, spaces.Discrete), \
            "DiscreteToBoxWrapper requires Discrete action space"
        
        self.n_actions = env.action_space.n
        
        # Create Box action space: single dimension [0, n_actions-1]
        self.action_space = spaces.Box(
            low=0.0,
            high=float(self.n_actions - 1),
            shape=(1,),
            dtype=np.float32
        )
    
    def action(self, action: np.ndarray) -> int:
        """
        Convert continuous action to discrete.
        
        Args:
            action: Continuous action array of shape (1,)
        
        Returns:
            Discrete action (integer)
        """
        # Clip and round to nearest integer
        continuous_action = float(action[0])
        discrete_action = int(np.clip(np.round(continuous_action), 0, self.n_actions - 1))
        return discrete_action


class DiscreteToMultiBinaryWrapper(gym.ActionWrapper):
    """
    Wrapper that converts a Discrete action space to MultiBinary.
    
    Uses one-hot encoding: action i -> [0, 0, ..., 1, ..., 0]
    The agent outputs probabilities, we select argmax.
    """
    
    def __init__(self, env: gym.Env):
        """
        Initialize the wrapper.
        
        Args:
            env: Environment with Discrete action space
        """
        super().__init__(env)
        
        assert isinstance(env.action_space, spaces.Discrete), \
            "DiscreteToMultiBinaryWrapper requires Discrete action space"
        
        self.n_actions = env.action_space.n
        
        # Create Box action space for logits/probabilities
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_actions,),
            dtype=np.float32
        )
    
    def action(self, action: np.ndarray) -> int:
        """
        Convert probability/logit vector to discrete action.
        
        Args:
            action: Array of shape (n_actions,) with logits or probabilities
        
        Returns:
            Discrete action (integer) - argmax of input
        """
        return int(np.argmax(action))


class NormalizedObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that normalizes observations using running statistics.
    """
    
    def __init__(self, env: gym.Env, clip: float = 10.0):
        """
        Initialize the wrapper.
        
        Args:
            env: Environment to wrap
            clip: Value to clip normalized observations to
        """
        super().__init__(env)
        self.clip = clip
        
        # Running statistics
        self.obs_mean = np.zeros(env.observation_space.shape)
        self.obs_var = np.ones(env.observation_space.shape)
        self.count = 0
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize observation using running statistics.
        
        Args:
            observation: Raw observation
        
        Returns:
            Normalized observation
        """
        # Update running statistics
        self.count += 1
        delta = observation - self.obs_mean
        self.obs_mean += delta / self.count
        self.obs_var += delta * (observation - self.obs_mean)
        
        # Normalize
        std = np.sqrt(self.obs_var / max(self.count, 1)) + 1e-8
        normalized = (observation - self.obs_mean) / std
        
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)


def wrap_env_for_sac(env: gym.Env, wrapper_type: str = "box") -> gym.Env:
    """
    Wrap environment for SAC training.
    
    Args:
        env: Environment with Discrete action space
        wrapper_type: Type of wrapper ("box" or "multibinary")
    
    Returns:
        Wrapped environment with continuous action space
    """
    if wrapper_type == "box":
        return DiscreteToBoxWrapper(env)
    elif wrapper_type == "multibinary":
        return DiscreteToMultiBinaryWrapper(env)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")
