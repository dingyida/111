"""Training Package for AoI RL."""

from training.baselines import (
    BaselinePolicy,
    RandomPolicy,
    RoundRobinPolicy,
    GreedyPoseAoIPolicy,
    GreedyStyleAoIPolicy,
    AdaptivePolicy,
    LeastRecentCameraPolicy,
    HighestAoIObjectPolicy,
    get_all_baselines,
    evaluate_baseline,
    compare_baselines
)
from training.train_ppo import AoITrainer
from training.train_sac import AoISACTrainer

__all__ = [
    'BaselinePolicy',
    'RandomPolicy',
    'RoundRobinPolicy',
    'GreedyPoseAoIPolicy',
    'GreedyStyleAoIPolicy',
    'AdaptivePolicy',
    'LeastRecentCameraPolicy',
    'HighestAoIObjectPolicy',
    'get_all_baselines',
    'evaluate_baseline',
    'compare_baselines',
    'AoITrainer',
    'AoISACTrainer'
]
