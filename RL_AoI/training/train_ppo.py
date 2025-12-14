#!/usr/bin/env python3
"""
PPO Training Script for AoI Minimization.

Uses Stable-Baselines3 to train a PPO agent.
"""

import os
import sys
import yaml
import argparse
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

import gymnasium as gym
from gymnasium import spaces

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from env.aoi_env import AoIEnv, ActionMaskedAoIEnv
from env.data_loader import PoseDataLoader


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(config: Dict, rank: int = 0, seed: int = 0):
    """
    Create a single environment instance.
    
    Args:
        config: Configuration dictionary
        rank: Environment rank for seeding
        seed: Base random seed
    
    Returns:
        Function that creates the environment
    """
    def _init():
        env = AoIEnv(config=config)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


class AoITrainer:
    """
    Trainer class for AoI minimization with PPO.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        log_dir: str = "./logs",
        model_dir: str = "./saved_models"
    ):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to config YAML file
            config: Configuration dictionary (overrides config_path)
            log_dir: Directory for logs
            model_dir: Directory for saved models
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            # Load default config file
            default_config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config', 'config.yaml'
            )
            if os.path.exists(default_config_path):
                self.config = load_config(default_config_path)
                print(f"Loaded config from {default_config_path}")
            else:
                # Fall back to hardcoded defaults
                self.config = self._default_config()
                print("Using hardcoded default config")
        
        # Set up directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"ppo_{timestamp}")
        self.model_dir = os.path.join(model_dir, f"ppo_{timestamp}")
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.log_dir, "config.yaml"), 'w') as f:
            yaml.dump(self.config, f)
        
        # Create environments
        self.env = None
        self.eval_env = None
        self.model = None
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'env': {
                'num_cameras': 5,
                'num_objects': 5,
                'episode_length': 200,
                'pose_aoi_increment': [0.006, 0.02, 0.05, 0.01, 0.036],
                'initial_pose_aoi': 0.0,
                'initial_style_aoi': 1.0,
                'resolutions': {
                    1: {'cooldown': 1, 'width': 640, 'height': 360},
                    2: {'cooldown': 4, 'width': 1280, 'height': 720}
                }
            },
            'reward': {
                'alpha': 0.5,
                'reward_scale': 1.0,
                'pose_aoi_max_expected': 10.0
            },
            'style_error': {
                'k': 4.0, 'p': 8.0, 'alpha': 0.2,
                'beta': 0.6, 'zeta': 1.0,
                'default_fov_deg': 45.0, 'n_samples': 1024
            },
            'data': {
                'render_output_path': './data/render_output',
                'variance_model_path': './models/variance_predictor_final.pth'
            },
            'ppo': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.02,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'policy_kwargs': {
                    'net_arch': {'pi': [256, 256], 'vf': [256, 256]}
                }
            },
            'training': {
                'total_timesteps': 500000,
                'num_envs': 4,
                'save_freq': 10000,
                'eval_freq': 5000,
                'augmentation': {
                    'random_start_frame': True,
                    'max_start_frame': 50,
                    'pose_noise_std': 0.001
                }
            }
        }
    
    def setup_environments(self, num_envs: Optional[int] = None):
        """
        Set up training and evaluation environments.
        
        Args:
            num_envs: Number of parallel environments
        """
        if num_envs is None:
            num_envs = self.config['training'].get('num_envs', 4)
        
        # Create vectorized training environment
        self.env = make_vec_env(
            make_env(self.config, 0, 42),
            n_envs=num_envs,
            vec_env_cls=DummyVecEnv
        )
        
        # Create evaluation environment
        eval_env = AoIEnv(config=self.config)
        self.eval_env = Monitor(eval_env)
    
    def create_model(self, pretrained_path: Optional[str] = None):
        """
        Create or load PPO model.
        
        Args:
            pretrained_path: Path to pretrained model (optional)
        """
        ppo_config = self.config['ppo']
        
        # Build policy kwargs
        policy_kwargs = {}
        if 'policy_kwargs' in ppo_config:
            net_arch = ppo_config['policy_kwargs'].get('net_arch')
            if net_arch:
                policy_kwargs['net_arch'] = [
                    dict(pi=net_arch.get('pi', [256, 256]),
                         vf=net_arch.get('vf', [256, 256]))
                ]
        
        if pretrained_path is not None and os.path.exists(pretrained_path):
            print(f"Loading pretrained model from {pretrained_path}")
            self.model = PPO.load(pretrained_path, env=self.env)
        else:
            print("Creating new PPO model")
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=ppo_config.get('learning_rate', 3e-4),
                n_steps=ppo_config.get('n_steps', 2048),
                batch_size=ppo_config.get('batch_size', 64),
                n_epochs=ppo_config.get('n_epochs', 10),
                gamma=ppo_config.get('gamma', 0.99),
                gae_lambda=ppo_config.get('gae_lambda', 0.95),
                clip_range=ppo_config.get('clip_range', 0.2),
                ent_coef=ppo_config.get('ent_coef', 0.02),
                vf_coef=ppo_config.get('vf_coef', 0.5),
                max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
                policy_kwargs=policy_kwargs if policy_kwargs else None,
                verbose=1,
                tensorboard_log=self.log_dir
            )
    
    def setup_callbacks(self) -> CallbackList:
        """
        Set up training callbacks.
        
        Returns:
            CallbackList with all callbacks
        """
        callbacks = []
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(self.model_dir, "best"),
            log_path=self.log_dir,
            eval_freq=self.config['training'].get('eval_freq', 5000),
            n_eval_episodes=10,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['training'].get('save_freq', 10000),
            save_path=os.path.join(self.model_dir, "checkpoints"),
            name_prefix="ppo_aoi"
        )
        callbacks.append(checkpoint_callback)
        
        return CallbackList(callbacks)
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        pretrained_path: Optional[str] = None
    ):
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total training timesteps
            pretrained_path: Path to pretrained model
        """
        if total_timesteps is None:
            total_timesteps = self.config['training'].get('total_timesteps', 500000)
        
        # Setup
        print("Setting up environments...")
        self.setup_environments()
        
        print("Creating model...")
        self.create_model(pretrained_path)
        
        print("Setting up callbacks...")
        callbacks = self.setup_callbacks()
        
        # Configure logger
        new_logger = configure(self.log_dir, ["stdout", "tensorboard"])
        self.model.set_logger(new_logger)
        
        # Train
        print(f"\nStarting training for {total_timesteps} timesteps...")
        print(f"Logs: {self.log_dir}")
        print(f"Models: {self.model_dir}")
        print("=" * 60)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(self.model_dir, "final_model")
        self.model.save(final_path)
        print(f"\nTraining complete! Final model saved to {final_path}")
    
    def evaluate(
        self,
        model_path: Optional[str] = None,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to model (uses current model if None)
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions
        
        Returns:
            Evaluation results dictionary
        """
        if model_path is not None:
            model = PPO.load(model_path)
        else:
            model = self.model
        
        if model is None:
            raise ValueError("No model available for evaluation")
        
        # Create evaluation environment
        eval_env = AoIEnv(config=self.config)
        
        episode_rewards = []
        episode_lengths = []
        episode_total_aoi = []
        
        for ep in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            total_reward = 0
            total_aoi_sum = 0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                total_reward += reward
                total_aoi_sum += info['total_aoi']
                steps += 1
                done = terminated or truncated
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_total_aoi.append(total_aoi_sum / steps)
            
            print(f"Episode {ep + 1}: reward={total_reward:.2f}, "
                  f"avg_aoi={total_aoi_sum/steps:.4f}, steps={steps}")
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_aoi': np.mean(episode_total_aoi),
            'episode_rewards': episode_rewards,
            'episode_aoi': episode_total_aoi
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean AoI: {results['mean_aoi']:.4f}")
        
        return results
    
    def close(self):
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train PPO for AoI Minimization")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Alpha parameter for reward (pose vs style weight)"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./saved_models",
        help="Directory for saved models"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only run evaluation (requires --pretrained)"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = None
    
    # Create trainer
    trainer = AoITrainer(
        config=config,
        log_dir=args.log_dir,
        model_dir=args.model_dir
    )
    
    # Override alpha if specified
    if args.alpha is not None:
        trainer.config['reward']['alpha'] = args.alpha
    
    try:
        if args.eval_only:
            if args.pretrained is None:
                raise ValueError("--pretrained required for evaluation")
            trainer.setup_environments()
            trainer.evaluate(model_path=args.pretrained)
        else:
            trainer.train(
                total_timesteps=args.timesteps,
                pretrained_path=args.pretrained
            )
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
