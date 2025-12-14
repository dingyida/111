#!/usr/bin/env python3
"""
SAC (Soft Actor-Critic) Training for AoI Minimization.

SAC is an off-policy algorithm that:
- Uses entropy regularization for exploration
- Learns a stochastic policy
- Often more sample efficient than PPO

Note: SAC requires continuous action space, so we wrap the discrete
environment with DiscreteToBoxWrapper.
"""

import os
import yaml
import numpy as np
from typing import Dict, Optional, Callable
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env.aoi_env import AoIEnv
from env.wrappers import DiscreteToBoxWrapper


def make_env(config: Optional[Dict] = None, rank: int = 0, seed: int = 0) -> Callable:
    """Create a function that creates a wrapped environment for SAC."""
    def _init() -> DiscreteToBoxWrapper:
        env = AoIEnv(config=config)
        env.reset(seed=seed + rank)
        # Wrap for continuous action space
        env = DiscreteToBoxWrapper(env)
        return Monitor(env)
    return _init


class AoISACTrainer:
    """
    SAC Trainer for AoI Minimization.
    
    SAC uses continuous action space by default, but we use a discrete
    action wrapper or treat discrete actions as continuous.
    """
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        config: Optional[Dict] = None
    ):
        """
        Initialize the SAC trainer.
        
        Args:
            config_path: Path to configuration file
            config: Optional config dict (overrides file)
        """
        if config is not None:
            self.config = config
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        self.model = None
        self.env = None
    
    def create_env(self, num_envs: int = 1, use_subprocess: bool = False) -> DummyVecEnv:
        """
        Create vectorized environment.
        
        Args:
            num_envs: Number of parallel environments
            use_subprocess: Whether to use subprocess vectorization
        
        Returns:
            Vectorized environment
        """
        env_fns = [make_env(self.config, rank=i) for i in range(num_envs)]
        
        if use_subprocess and num_envs > 1:
            return SubprocVecEnv(env_fns)
        else:
            return DummyVecEnv(env_fns)
    
    def train(
        self,
        total_timesteps: int = 500000,
        num_envs: int = 1,
        save_freq: int = 10000,
        eval_freq: int = 5000,
        log_dir: str = "./logs",
        model_dir: str = "./saved_models",
        alpha: Optional[float] = None,
        pretrained_path: Optional[str] = None
    ):
        """
        Train SAC agent.
        
        Args:
            total_timesteps: Total training timesteps
            num_envs: Number of parallel environments (SAC typically uses 1)
            save_freq: Checkpoint save frequency
            eval_freq: Evaluation frequency
            log_dir: Tensorboard log directory
            model_dir: Model save directory
            alpha: Optional alpha override for reward
            pretrained_path: Path to pretrained model to continue training
        """
        # Override alpha if specified
        if alpha is not None:
            self.config['reward']['alpha'] = alpha
        
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"sac_aoi_{timestamp}"
        
        log_path = os.path.join(log_dir, run_name)
        model_path = os.path.join(model_dir, run_name)
        best_model_path = os.path.join(model_dir, "best_sac")
        
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(best_model_path, exist_ok=True)
        
        # Create environments
        # SAC is off-policy, typically uses single env
        self.env = self.create_env(num_envs=num_envs)
        eval_env = self.create_env(num_envs=1)
        
        # Get SAC hyperparameters
        sac_config = self.config.get('sac', {})
        
        # Create or load model
        if pretrained_path is not None and os.path.exists(pretrained_path):
            print(f"Loading pretrained model from {pretrained_path}")
            self.model = SAC.load(pretrained_path, env=self.env)
        else:
            # SAC with discrete actions requires special handling
            # We use MultiDiscrete or just Discrete with proper wrapper
            self.model = SAC(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=sac_config.get('learning_rate', 3e-4),
                buffer_size=sac_config.get('buffer_size', 100000),
                learning_starts=sac_config.get('learning_starts', 1000),
                batch_size=sac_config.get('batch_size', 256),
                tau=sac_config.get('tau', 0.005),
                gamma=sac_config.get('gamma', 0.99),
                train_freq=sac_config.get('train_freq', 1),
                gradient_steps=sac_config.get('gradient_steps', 1),
                ent_coef=sac_config.get('ent_coef', 'auto'),
                target_entropy=sac_config.get('target_entropy', 'auto'),
                use_sde=sac_config.get('use_sde', False),
                policy_kwargs=sac_config.get('policy_kwargs', {
                    'net_arch': [256, 256]
                }),
                verbose=1,
                tensorboard_log=log_path
            )
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq // num_envs,
            save_path=model_path,
            name_prefix="sac_aoi"
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_path,
            log_path=log_path,
            eval_freq=eval_freq // num_envs,
            n_eval_episodes=5,
            deterministic=True
        )
        
        # Train
        print(f"\n{'='*60}")
        print(f"Starting SAC Training")
        print(f"{'='*60}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Num envs: {num_envs}")
        print(f"Alpha: {self.config['reward']['alpha']}")
        print(f"Log dir: {log_path}")
        print(f"Model dir: {model_path}")
        print(f"{'='*60}\n")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(model_path, "final_model")
        self.model.save(final_path)
        print(f"\nTraining complete! Final model saved to {final_path}")
        
        return self.model
    
    def load(self, model_path: str):
        """Load a trained model."""
        self.model = SAC.load(model_path)
        return self.model
    
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions
        
        Returns:
            Evaluation results
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        env = AoIEnv(config=self.config)
        env = DiscreteToBoxWrapper(env)
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                length += 1
                done = terminated or truncated
            
            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            print(f"Episode {ep+1}: reward={total_reward:.2f}, length={length}")
        
        env.close()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards
        }


def train_sac(
    config_path: str = "config/config.yaml",
    total_timesteps: int = 500000,
    alpha: Optional[float] = None
):
    """Convenience function to train SAC."""
    trainer = AoISACTrainer(config_path=config_path)
    return trainer.train(total_timesteps=total_timesteps, alpha=alpha)


if __name__ == "__main__":
    train_sac()
