#!/usr/bin/env python3
"""
Baseline Policies for AoI Minimization.

Provides simple heuristic policies for comparison with learned RL policy.
"""

import numpy as np
from typing import Optional, List
from abc import ABC, abstractmethod


class BaselinePolicy(ABC):
    """Abstract base class for baseline policies."""
    
    def __init__(self, num_cameras: int = 5, num_resolutions: int = 2):
        self.num_cameras = num_cameras
        self.num_resolutions = num_resolutions
        self.num_actions = num_cameras * num_resolutions
    
    @abstractmethod
    def get_action(self, observation: np.ndarray, info: dict) -> int:
        """
        Get action based on current observation.
        
        Args:
            observation: Current observation vector
            info: Additional info dict from environment
        
        Returns:
            Action (0-9)
        """
        pass
    
    def reset(self):
        """Reset policy state (if any)."""
        pass
    
    @property
    def name(self) -> str:
        """Return policy name."""
        return self.__class__.__name__


class RandomPolicy(BaselinePolicy):
    """Random action selection."""
    
    def __init__(self, num_cameras: int = 5, num_resolutions: int = 2, seed: int = 42):
        super().__init__(num_cameras, num_resolutions)
        self.rng = np.random.default_rng(seed)
    
    def get_action(self, observation: np.ndarray, info: dict) -> int:
        return self.rng.integers(0, self.num_actions)


class RoundRobinPolicy(BaselinePolicy):
    """Cycle through cameras with fixed resolution."""
    
    def __init__(
        self,
        num_cameras: int = 5,
        num_resolutions: int = 2,
        use_high_res: bool = False
    ):
        super().__init__(num_cameras, num_resolutions)
        self.use_high_res = use_high_res
        self.current_camera = 0
    
    def get_action(self, observation: np.ndarray, info: dict) -> int:
        # Select current camera with chosen resolution
        if self.use_high_res:
            action = self.current_camera + 5  # Resolution 2
        else:
            action = self.current_camera  # Resolution 1
        
        # Move to next camera
        self.current_camera = (self.current_camera + 1) % self.num_cameras
        
        return action
    
    def reset(self):
        self.current_camera = 0


class GreedyPoseAoIPolicy(BaselinePolicy):
    """
    Greedy policy that prioritizes reducing high pose AoI.
    
    Always uses low resolution (more frequent observations).
    Selects camera that hasn't been used recently.
    """
    
    def __init__(self, num_cameras: int = 5, num_resolutions: int = 2):
        super().__init__(num_cameras, num_resolutions)
        self.last_camera = -1
    
    def get_action(self, observation: np.ndarray, info: dict) -> int:
        # Cycle through cameras, avoiding the last used one
        camera = (self.last_camera + 1) % self.num_cameras
        self.last_camera = camera
        
        # Use low resolution for more frequent updates
        return camera  # Actions 0-4 are resolution 1
    
    def reset(self):
        self.last_camera = -1


class GreedyStyleAoIPolicy(BaselinePolicy):
    """
    Greedy policy that prioritizes reducing style AoI.
    
    Uses high resolution for better style coverage.
    Cycles through cameras to maximize viewpoint diversity.
    """
    
    def __init__(self, num_cameras: int = 5, num_resolutions: int = 2):
        super().__init__(num_cameras, num_resolutions)
        self.current_camera = 0
    
    def get_action(self, observation: np.ndarray, info: dict) -> int:
        # Use high resolution for better style coverage
        action = self.current_camera + 5  # Actions 5-9 are resolution 2
        
        # Cycle through cameras
        self.current_camera = (self.current_camera + 1) % self.num_cameras
        
        return action
    
    def reset(self):
        self.current_camera = 0


class AdaptivePolicy(BaselinePolicy):
    """
    Adaptive policy that switches between pose and style focus.
    
    - If pose AoI is high → use low resolution (frequent updates)
    - If style AoI is high → use high resolution (better coverage)
    """
    
    def __init__(
        self,
        num_cameras: int = 5,
        num_resolutions: int = 2,
        pose_threshold: float = 0.5,
        style_threshold: float = 0.7
    ):
        super().__init__(num_cameras, num_resolutions)
        self.pose_threshold = pose_threshold
        self.style_threshold = style_threshold
        self.current_camera = 0
    
    def get_action(self, observation: np.ndarray, info: dict) -> int:
        # Extract AoI values from observation
        # observation[0:5] = pose_aoi_normalized
        # observation[5:10] = style_aoi
        pose_aoi = observation[:5]
        style_aoi = observation[5:10]
        
        avg_pose = np.mean(pose_aoi)
        avg_style = np.mean(style_aoi)
        
        # Decide resolution based on current AoI levels
        if avg_pose > self.pose_threshold:
            # High pose AoI → use low resolution for frequent updates
            action = self.current_camera
        elif avg_style > self.style_threshold:
            # High style AoI → use high resolution for better coverage
            action = self.current_camera + 5
        else:
            # Balanced → alternate
            if self.current_camera % 2 == 0:
                action = self.current_camera
            else:
                action = self.current_camera + 5
        
        self.current_camera = (self.current_camera + 1) % self.num_cameras
        return action
    
    def reset(self):
        self.current_camera = 0


class LeastRecentCameraPolicy(BaselinePolicy):
    """
    Select the camera that was used least recently.
    
    Helps maximize viewpoint diversity for style coverage.
    """
    
    def __init__(
        self,
        num_cameras: int = 5,
        num_resolutions: int = 2,
        use_high_res: bool = False
    ):
        super().__init__(num_cameras, num_resolutions)
        self.use_high_res = use_high_res
        self.camera_last_used = np.zeros(num_cameras)
        self.step = 0
    
    def get_action(self, observation: np.ndarray, info: dict) -> int:
        # Find camera with smallest last-used time
        camera = np.argmin(self.camera_last_used)
        
        # Update last-used time
        self.step += 1
        self.camera_last_used[camera] = self.step
        
        # Select resolution
        if self.use_high_res:
            return camera + 5
        else:
            return camera
    
    def reset(self):
        self.camera_last_used = np.zeros(self.num_cameras)
        self.step = 0


class HighestAoIObjectPolicy(BaselinePolicy):
    """
    Focus on reducing AoI for the object with highest current AoI.
    
    Note: Since all cameras observe all objects, this mainly affects
    the timing of observations.
    """
    
    def __init__(
        self,
        num_cameras: int = 5,
        num_resolutions: int = 2,
        alpha: float = 0.5
    ):
        super().__init__(num_cameras, num_resolutions)
        self.alpha = alpha
        self.current_camera = 0
    
    def get_action(self, observation: np.ndarray, info: dict) -> int:
        # Compute combined AoI for each object
        pose_aoi = observation[:5]
        style_aoi = observation[5:10]
        combined_aoi = self.alpha * pose_aoi + (1 - self.alpha) * style_aoi
        
        # Find object with highest AoI
        max_obj = np.argmax(combined_aoi)
        
        # If pose AoI dominates → low res, else high res
        if pose_aoi[max_obj] > style_aoi[max_obj]:
            action = self.current_camera
        else:
            action = self.current_camera + 5
        
        self.current_camera = (self.current_camera + 1) % self.num_cameras
        return action
    
    def reset(self):
        self.current_camera = 0


def get_all_baselines() -> List[BaselinePolicy]:
    """Get list of all baseline policies."""
    return [
        RandomPolicy(),
        RoundRobinPolicy(use_high_res=False),
        RoundRobinPolicy(use_high_res=True),
        GreedyPoseAoIPolicy(),
        GreedyStyleAoIPolicy(),
        AdaptivePolicy(),
        LeastRecentCameraPolicy(use_high_res=False),
        LeastRecentCameraPolicy(use_high_res=True),
        HighestAoIObjectPolicy()
    ]


def evaluate_baseline(
    env,
    policy: BaselinePolicy,
    n_episodes: int = 10,
    verbose: bool = False
) -> dict:
    """
    Evaluate a baseline policy.
    
    Args:
        env: AoI environment
        policy: Baseline policy to evaluate
        n_episodes: Number of episodes to run
        verbose: Whether to print progress
    
    Returns:
        Dictionary with evaluation results
    """
    episode_rewards = []
    episode_pose_aoi = []
    episode_style_aoi = []
    episode_total_aoi = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        policy.reset()
        
        total_reward = 0
        pose_aoi_sum = 0
        style_aoi_sum = 0
        total_aoi_sum = 0
        steps = 0
        
        done = False
        while not done:
            # Skip action if in cooldown
            if info.get('can_act', True):
                action = policy.get_action(obs, info)
            else:
                action = 0  # Doesn't matter, will be ignored
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            pose_aoi_sum += info['total_pose_aoi']
            style_aoi_sum += info['total_style_aoi']
            total_aoi_sum += info['total_aoi']
            steps += 1
            
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        episode_pose_aoi.append(pose_aoi_sum / steps)
        episode_style_aoi.append(style_aoi_sum / steps)
        episode_total_aoi.append(total_aoi_sum / steps)
        
        if verbose:
            print(f"Episode {ep + 1}: reward={total_reward:.2f}, "
                  f"avg_total_aoi={total_aoi_sum/steps:.4f}")
    
    return {
        'policy': policy.name,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_pose_aoi': np.mean(episode_pose_aoi),
        'mean_style_aoi': np.mean(episode_style_aoi),
        'mean_total_aoi': np.mean(episode_total_aoi),
        'episode_rewards': episode_rewards
    }


def compare_baselines(env, n_episodes: int = 10) -> List[dict]:
    """
    Compare all baseline policies.
    
    Args:
        env: AoI environment
        n_episodes: Number of episodes per policy
    
    Returns:
        List of evaluation results
    """
    baselines = get_all_baselines()
    results = []
    
    print("Comparing baseline policies...")
    print("=" * 60)
    
    for policy in baselines:
        print(f"\nEvaluating {policy.name}...")
        result = evaluate_baseline(env, policy, n_episodes)
        results.append(result)
        
        print(f"  Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Mean total AoI: {result['mean_total_aoi']:.4f}")
    
    # Sort by mean reward (higher is better, i.e., less negative)
    results.sort(key=lambda x: x['mean_reward'], reverse=True)
    
    print("\n" + "=" * 60)
    print("Rankings (by mean reward):")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['policy']}: {r['mean_reward']:.2f}")
    
    return results


if __name__ == "__main__":
    # Test baselines
    import sys
    sys.path.insert(0, '/home/claude/aoi_rl_project')
    
    from env.aoi_env import AoIEnv
    
    env = AoIEnv()
    results = compare_baselines(env, n_episodes=5)
