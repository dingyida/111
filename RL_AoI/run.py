#!/usr/bin/env python3
"""
Main Entry Point for AoI RL Project.

Usage:
    # 1. Compare PPO model with baselines
    python run.py compare --model ./saved_models/best/best_model.zip
    
    # 2. Evaluate PPO model and output all decisions
    python run.py eval --model ./saved_models/best/best_model.zip
    
    # 3. Train PPO with reward logging (saves CSV with rewards over time)
    python run.py train --timesteps 500000
    
    # Other commands
    python run.py train --model-type sac --timesteps 500000
    python run.py baselines
    python run.py test-env
"""

import os
import sys
import argparse
import csv
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


# =============================================================================
# 1. COMPARE WITH BASELINES
# =============================================================================

def cmd_compare(args):
    """
    Compare trained PPO/SAC model with baseline policies.
    
    Outputs a table showing performance of:
    - Trained model (PPO or SAC)
    - Random policy
    - Round-robin policy
    - Greedy pose AoI policy
    - Greedy style AoI policy
    - Adaptive policy
    """
    from stable_baselines3 import PPO, SAC
    from env.aoi_env import AoIEnv
    from env.wrappers import DiscreteToBoxWrapper
    from training.baselines import get_all_baselines, evaluate_baseline
    
    print("=" * 90)
    print("BASELINE COMPARISON")
    print("=" * 90)
    
    # Load model
    model_type = args.model_type.lower()
    if model_type == 'ppo':
        model = PPO.load(args.model)
    elif model_type == 'sac':
        model = SAC.load(args.model)
    else:
        print(f"Unknown model type: {model_type}")
        return
    
    # Create environment
    env = AoIEnv()
    
    # Evaluate trained model
    print(f"\nEvaluating {model_type.upper()} model...")
    model_results = _evaluate_policy(
        env, model, args.episodes, 
        is_sac=(model_type == 'sac'),
        name=f"{model_type.upper()} Model"
    )
    
    # Evaluate baselines
    baselines = get_all_baselines()
    baseline_results = []
    
    for policy in baselines:
        print(f"Evaluating {policy.name}...")
        results = evaluate_baseline(env, policy, n_episodes=args.episodes)
        baseline_results.append({
            'name': policy.name,
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'mean_pose_aoi': results.get('mean_pose_aoi', 0),
            'mean_style_aoi': results.get('mean_style_aoi', 0)
        })
    
    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON RESULTS")
    print("=" * 90)
    print(f"\n{'Policy':<25} {'Mean Reward':<15} {'Std':<10} {'Pose AoI':<12} {'Style AoI':<12}")
    print("-" * 90)
    
    # Print trained model first
    print(f"{model_type.upper() + ' Model':<25} {model_results['mean_reward']:<15.2f} "
          f"{model_results['std_reward']:<10.2f} {model_results['mean_pose_aoi']:<12.4f} "
          f"{model_results['mean_style_aoi']:<12.4f}")
    
    # Print baselines
    for r in baseline_results:
        print(f"{r['name']:<25} {r['mean_reward']:<15.2f} {r['std_reward']:<10.2f} "
              f"{r['mean_pose_aoi']:<12.4f} {r['mean_style_aoi']:<12.4f}")
    
    print("-" * 90)
    
    # Calculate improvement over best baseline
    best_baseline_reward = max(r['mean_reward'] for r in baseline_results)
    improvement = model_results['mean_reward'] - best_baseline_reward
    pct_improvement = (improvement / abs(best_baseline_reward)) * 100 if best_baseline_reward != 0 else 0
    
    print(f"\n{model_type.upper()} vs Best Baseline: {improvement:+.2f} ({pct_improvement:+.1f}%)")
    
    # Save results to file
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write("Policy,Mean Reward,Std,Pose AoI,Style AoI\n")
            f.write(f"{model_type.upper()} Model,{model_results['mean_reward']:.4f},"
                    f"{model_results['std_reward']:.4f},{model_results['mean_pose_aoi']:.4f},"
                    f"{model_results['mean_style_aoi']:.4f}\n")
            for r in baseline_results:
                f.write(f"{r['name']},{r['mean_reward']:.4f},{r['std_reward']:.4f},"
                        f"{r['mean_pose_aoi']:.4f},{r['mean_style_aoi']:.4f}\n")
        print(f"\nResults saved to {args.output}")
    
    env.close()


def _evaluate_policy(env, model, n_episodes, is_sac=False, name="Model"):
    """Helper to evaluate a trained model."""
    from env.wrappers import DiscreteToBoxWrapper
    
    episode_rewards = []
    episode_pose_aoi = []
    episode_style_aoi = []
    
    for ep in range(n_episodes):
        if is_sac:
            # SAC needs wrapped env
            wrapped_env = DiscreteToBoxWrapper(env)
            obs, info = wrapped_env.reset()
        else:
            obs, info = env.reset()
        
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if is_sac:
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        episode_pose_aoi.append(info['total_pose_aoi'])
        episode_style_aoi.append(info['total_style_aoi'])
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_pose_aoi': np.mean(episode_pose_aoi),
        'mean_style_aoi': np.mean(episode_style_aoi)
    }


# =============================================================================
# 2. EVALUATE AND OUTPUT ALL DECISIONS
# =============================================================================

def cmd_eval(args):
    """
    Evaluate PPO/SAC model and output all decisions in order.
    
    For each episode, outputs a detailed table showing:
    - Frame number
    - Decision (camera + resolution)
    - Reward
    - Pose contribution
    - Style contribution
    - Total AoI
    """
    from stable_baselines3 import PPO, SAC
    from env.aoi_env import AoIEnv
    from env.wrappers import DiscreteToBoxWrapper
    
    print("=" * 100)
    print("PPO MODEL EVALUATION - ALL DECISIONS")
    print("=" * 100)
    
    # Load model
    model_type = args.model_type.lower()
    if model_type == 'ppo':
        model = PPO.load(args.model)
    elif model_type == 'sac':
        model = SAC.load(args.model)
    
    # Create environment
    env = AoIEnv()
    
    # Store all decisions for CSV output
    all_decisions = []
    episode_summaries = []
    
    for ep in range(args.episodes):
        print(f"\n{'='*100}")
        print(f"Episode {ep + 1} - Detailed Step Output")
        print("=" * 100)
        
        # Header
        print(f"\n{'Frame':<7} {'Decision':<20} {'Reward':<12} {'Pose Contrib':<14} "
              f"{'Style Contrib':<14} {'Total AoI':<12} {'Obs?':<6}")
        print("-" * 100)
        
        if model_type == 'sac':
            wrapped_env = DiscreteToBoxWrapper(env)
            obs, info = wrapped_env.reset()
        else:
            obs, info = env.reset()
        
        done = False
        total_reward = 0
        total_pose_contrib = 0
        total_style_contrib = 0
        step = 0
        episode_decisions = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            if model_type == 'sac':
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            total_pose_contrib += info['pose_contribution']
            total_style_contrib += info['style_contribution']
            step += 1
            done = terminated or truncated
            
            # Format decision
            decision = info['decision']
            decision_str = f"Cam{decision['camera_id']}-{decision['resolution_name']}"
            if not decision['action_taken']:
                decision_str += " (blocked)"
            
            obs_str = "Yes" if info.get('observation_completed', False) else ""
            
            # Print row
            print(f"{info['frame']:<7} {decision_str:<20} {reward:<12.4f} "
                  f"{info['pose_contribution']:<14.4f} {info['style_contribution']:<14.4f} "
                  f"{info['total_aoi']:<12.4f} {obs_str:<6}")
            
            # Store decision
            episode_decisions.append({
                'episode': ep + 1,
                'frame': info['frame'],
                'camera_id': decision['camera_id'],
                'resolution': decision['resolution_name'],
                'action_taken': decision['action_taken'],
                'reward': reward,
                'pose_contribution': info['pose_contribution'],
                'style_contribution': info['style_contribution'],
                'total_aoi': info['total_aoi'],
                'pose_aoi': info['total_pose_aoi'],
                'style_aoi': info['total_style_aoi']
            })
        
        all_decisions.extend(episode_decisions)
        
        # Episode summary
        print("-" * 100)
        print(f"\nEpisode {ep + 1} Summary:")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Avg Pose Contribution: {total_pose_contrib/step:.4f}")
        print(f"  Avg Style Contribution: {total_style_contrib/step:.4f}")
        print(f"  Final Pose AoI: {info['total_pose_aoi']:.4f}")
        print(f"  Final Style AoI: {info['total_style_aoi']:.4f}")
        
        # Camera usage
        camera_counts = {}
        for d in episode_decisions:
            if d['action_taken']:
                key = f"Cam{d['camera_id']}-{d['resolution']}"
                camera_counts[key] = camera_counts.get(key, 0) + 1
        print(f"  Camera Usage: {camera_counts}")
        
        episode_summaries.append({
            'episode': ep + 1,
            'total_reward': total_reward,
            'avg_pose_contrib': total_pose_contrib / step,
            'avg_style_contrib': total_style_contrib / step,
            'final_pose_aoi': info['total_pose_aoi'],
            'final_style_aoi': info['total_style_aoi']
        })
    
    # Overall summary
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    
    rewards = [s['total_reward'] for s in episode_summaries]
    print(f"\nMean Reward: {np.mean(rewards):.4f} ¡À {np.std(rewards):.4f}")
    print(f"Mean Pose Contribution: {np.mean([s['avg_pose_contrib'] for s in episode_summaries]):.4f}")
    print(f"Mean Style Contribution: {np.mean([s['avg_style_contrib'] for s in episode_summaries]):.4f}")
    
    # Save decisions to CSV
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_decisions[0].keys())
            writer.writeheader()
            writer.writerows(all_decisions)
        print(f"\nAll decisions saved to {args.output}")
    
    env.close()


# =============================================================================
# 3. TRAIN WITH REWARD LOGGING
# =============================================================================

def cmd_train(args):
    """
    Train PPO/SAC agent with detailed reward logging.
    
    Saves a CSV file with columns:
    - timestep
    - total_reward
    - pose_reward (contribution)
    - style_reward (contribution)
    - episode_length
    """
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from env.aoi_env import AoIEnv
    from env.wrappers import DiscreteToBoxWrapper
    import yaml
    
    # Load config
    config_path = args.config if args.config else os.path.join(project_root, "config", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
    else:
        config = None
        print("Using default config")
    
    # Override alpha if specified
    if args.alpha is not None and config:
        config['reward']['alpha'] = args.alpha
        print(f"Using alpha = {args.alpha}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = args.model_type.lower()
    run_name = f"{model_type}_aoi_{timestamp}"
    
    log_dir = os.path.join(args.log_dir, run_name)
    model_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # CSV file for reward logging
    reward_log_path = os.path.join(log_dir, "reward_log.csv")
    
    # Custom callback for detailed reward logging
    class RewardLoggingCallback(BaseCallback):
        """Callback to log total/pose/style rewards to CSV."""
        
        def __init__(self, log_path, verbose=0):
            super().__init__(verbose)
            self.log_path = log_path
            self.episode_rewards = []
            self.episode_pose_contribs = []
            self.episode_style_contribs = []
            self.episode_lengths = []
            self.current_rewards = None
            self.current_pose = None
            self.current_style = None
            self.current_lengths = None
            
            # Initialize CSV
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestep', 'episode', 'total_reward', 'pose_reward', 
                                'style_reward', 'episode_length', 'mean_reward_100ep'])
        
        def _on_training_start(self):
            n_envs = self.training_env.num_envs
            self.current_rewards = np.zeros(n_envs)
            self.current_pose = np.zeros(n_envs)
            self.current_style = np.zeros(n_envs)
            self.current_lengths = np.zeros(n_envs)
        
        def _on_step(self):
            # Get info from environments
            infos = self.locals.get('infos', [])
            dones = self.locals.get('dones', [])
            rewards = self.locals.get('rewards', [])
            
            for i, (info, done, reward) in enumerate(zip(infos, dones, rewards)):
                self.current_rewards[i] += reward
                self.current_lengths[i] += 1
                
                # Get pose/style contributions from info
                if 'pose_contribution' in info:
                    self.current_pose[i] += info['pose_contribution']
                    self.current_style[i] += info['style_contribution']
                
                if done:
                    # Episode finished, log it
                    self.episode_rewards.append(self.current_rewards[i])
                    self.episode_pose_contribs.append(self.current_pose[i])
                    self.episode_style_contribs.append(self.current_style[i])
                    self.episode_lengths.append(self.current_lengths[i])
                    
                    # Calculate running mean
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    
                    # Write to CSV
                    with open(self.log_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            self.num_timesteps,
                            len(self.episode_rewards),
                            self.current_rewards[i],
                            self.current_pose[i],
                            self.current_style[i],
                            self.current_lengths[i],
                            mean_reward
                        ])
                    
                    # Reset for this env
                    self.current_rewards[i] = 0
                    self.current_pose[i] = 0
                    self.current_style[i] = 0
                    self.current_lengths[i] = 0
            
            return True
    
    # Create environments
    def make_env(rank=0):
        def _init():
            env = AoIEnv(config=config)
            env.reset(seed=42 + rank)
            if model_type == 'sac':
                env = DiscreteToBoxWrapper(env)
            return Monitor(env)
        return _init
    
    n_envs = args.num_envs if model_type == 'ppo' else 1
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    # Create eval environment
    eval_env = AoIEnv(config=config)
    if model_type == 'sac':
        eval_env = DiscreteToBoxWrapper(eval_env)
    eval_env = Monitor(eval_env)
    
    # Get hyperparameters from config
    if model_type == 'ppo':
        hp = config.get('ppo', {}) if config else {}
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=hp.get('learning_rate', 3e-4),
            n_steps=hp.get('n_steps', 2048),
            batch_size=hp.get('batch_size', 64),
            n_epochs=hp.get('n_epochs', 10),
            gamma=hp.get('gamma', 0.99),
            gae_lambda=hp.get('gae_lambda', 0.95),
            clip_range=hp.get('clip_range', 0.2),
            ent_coef=hp.get('ent_coef', 0.02),
            vf_coef=hp.get('vf_coef', 0.5),
            max_grad_norm=hp.get('max_grad_norm', 0.5),
            verbose=1,
            tensorboard_log=log_dir
        )
    else:  # SAC
        hp = config.get('sac', {}) if config else {}
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=hp.get('learning_rate', 3e-4),
            buffer_size=hp.get('buffer_size', 100000),
            learning_starts=hp.get('learning_starts', 1000),
            batch_size=hp.get('batch_size', 256),
            tau=hp.get('tau', 0.005),
            gamma=hp.get('gamma', 0.99),
            train_freq=hp.get('train_freq', 1),
            gradient_steps=hp.get('gradient_steps', 1),
            verbose=1,
            tensorboard_log=log_dir
        )
    
    # Setup callbacks
    callbacks = [
        RewardLoggingCallback(reward_log_path),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best"),
            log_path=log_dir,
            eval_freq=5000 // n_envs,
            n_eval_episodes=5,
            deterministic=True
        ),
        CheckpointCallback(
            save_freq=10000 // n_envs,
            save_path=os.path.join(model_dir, "checkpoints"),
            name_prefix=f"{model_type}_aoi"
        )
    ]
    
    # Print training info
    print("\n" + "=" * 60)
    print(f"TRAINING {model_type.upper()}")
    print("=" * 60)
    print(f"Timesteps: {args.timesteps}")
    print(f"Num envs: {n_envs}")
    print(f"Log dir: {log_dir}")
    print(f"Model dir: {model_dir}")
    print(f"Reward log: {reward_log_path}")
    print("=" * 60 + "\n")
    
    # Train
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final model: {final_path}")
    print(f"Best model: {os.path.join(model_dir, 'best', 'best_model.zip')}")
    print(f"Reward log: {reward_log_path}")
    print("=" * 60)
    
    env.close()
    eval_env.close()


# =============================================================================
# OTHER COMMANDS
# =============================================================================

def cmd_baselines(args):
    """Run baseline policies and compare them."""
    from env.aoi_env import AoIEnv
    from training.baselines import get_all_baselines, evaluate_baseline
    
    print("=" * 80)
    print("BASELINE EVALUATION")
    print("=" * 80)
    
    env = AoIEnv()
    baselines = get_all_baselines()
    
    print(f"\n{'Policy':<30} {'Mean Reward':<15} {'Std':<10} {'Pose AoI':<12} {'Style AoI':<12}")
    print("-" * 85)
    
    for policy in baselines:
        results = evaluate_baseline(env, policy, n_episodes=args.episodes)
        print(f"{policy.name:<30} {results['mean_reward']:<15.2f} {results['std_reward']:<10.2f} "
              f"{results.get('mean_pose_aoi', 0):<12.4f} {results.get('mean_style_aoi', 0):<12.4f}")
    
    print("-" * 80)
    env.close()


def cmd_test_env(args):
    """Test the environment and show config."""
    from env.aoi_env import AoIEnv
    
    print("=" * 80)
    print("ENVIRONMENT TEST")
    print("=" * 80)
    
    env = AoIEnv()
    obs, info = env.reset(seed=42)
    
    print(f"\nConfig source: {env._config_source}")
    print(f"\nObservation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    print("\n--- Resolution Config ---")
    print(f"Resolution 1 (640x360):  cooldown = {env.resolution_cooldowns[1]}, size = {env.resolution_sizes[1]}")
    print(f"Resolution 2 (1280x720): cooldown = {env.resolution_cooldowns[2]}, size = {env.resolution_sizes[2]}")
    print(f"[DEBUG] Raw: {env.config['env']['resolutions']}")
    
    print("\n--- Reward Config ---")
    print(f"alpha: {env.alpha}")
    print(f"pose_aoi_norm: {env.pose_aoi_norm}")
    print(f"style_aoi_norm: {env.style_aoi_norm}")
    
    print("\n--- Style Error Config ---")
    se = env.aoi_manager.style_error_params
    print(f"k={se.get('k')}, p={se.get('p')}, alpha={se.get('alpha')}, beta={se.get('beta')}, zeta={se.get('zeta')}")
    print(f"n_samples={se.get('n_samples')} (sphere has {len(env.aoi_manager.sphere_samples)} points)")
    
    print("\n--- Sample Run (10 steps) ---")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        decision = info['decision']
        print(f"Frame {info['frame']}: Cam{decision['camera_id']}-{decision['resolution_name']} "
              f"reward={reward:.4f}")
        if terminated:
            break
    
    env.close()
    print("\n" + "=" * 80)


def cmd_plot_rewards(args):
    """Plot rewards from training log CSV."""
    import matplotlib.pyplot as plt
    
    if not os.path.exists(args.log):
        print(f"Log file not found: {args.log}")
        return
    
    # Read CSV
    timesteps = []
    total_rewards = []
    pose_rewards = []
    style_rewards = []
    mean_rewards = []
    
    with open(args.log, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps.append(int(row['timestep']))
            total_rewards.append(float(row['total_reward']))
            pose_rewards.append(float(row['pose_reward']))
            style_rewards.append(float(row['style_reward']))
            mean_rewards.append(float(row['mean_reward_100ep']))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Total reward over episodes
    ax1 = axes[0, 0]
    ax1.plot(total_rewards, alpha=0.3, label='Episode Reward')
    ax1.plot(mean_rewards, linewidth=2, label='Mean (100 ep)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Reward Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Pose vs Style contribution
    ax2 = axes[0, 1]
    ax2.plot(pose_rewards, alpha=0.5, label='Pose Contribution')
    ax2.plot(style_rewards, alpha=0.5, label='Style Contribution')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Contribution')
    ax2.set_title('Pose vs Style Contribution Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reward vs timesteps
    ax3 = axes[1, 0]
    ax3.scatter(timesteps, total_rewards, alpha=0.2, s=5)
    ax3.plot(timesteps, mean_rewards, color='red', linewidth=2, label='Mean (100 ep)')
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Total Reward')
    ax3.set_title('Reward vs Timesteps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Pose/Style ratio
    ax4 = axes[1, 1]
    ratios = [p / (p + s) if (p + s) > 0 else 0.5 for p, s in zip(pose_rewards, style_rewards)]
    ax4.plot(ratios, alpha=0.3)
    # Rolling average
    window = min(100, len(ratios))
    rolling_ratio = np.convolve(ratios, np.ones(window)/window, mode='valid')
    ax4.plot(range(window-1, len(ratios)), rolling_ratio, linewidth=2, label=f'Rolling Avg ({window})')
    ax4.axhline(y=0.5, color='gray', linestyle='--', label='50%')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Pose Contribution Ratio')
    ax4.set_title('Pose / (Pose + Style) Ratio Over Training')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AoI Minimization RL Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # 1. Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model with baselines')
    compare_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    compare_parser.add_argument('--model-type', type=str, default='ppo', choices=['ppo', 'sac'])
    compare_parser.add_argument('--episodes', type=int, default=10, help='Episodes per policy')
    compare_parser.add_argument('--output', type=str, default='./results/comparison.csv', help='Output CSV')
    
    # 2. Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model and output all decisions')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    eval_parser.add_argument('--model-type', type=str, default='ppo', choices=['ppo', 'sac'])
    eval_parser.add_argument('--episodes', type=int, default=1, help='Number of episodes')
    eval_parser.add_argument('--output', type=str, default='./results/decisions.csv', help='Output CSV')
    
    # 3. Train command
    train_parser = subparsers.add_parser('train', help='Train with reward logging')
    train_parser.add_argument('--model-type', type=str, default='ppo', choices=['ppo', 'sac'])
    train_parser.add_argument('--config', type=str, help='Path to config file')
    train_parser.add_argument('--timesteps', type=int, default=500000, help='Training timesteps')
    train_parser.add_argument('--alpha', type=float, help='Alpha parameter')
    train_parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel envs')
    train_parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    train_parser.add_argument('--model-dir', type=str, default='./saved_models', help='Model directory')
    
    # Baselines command
    baselines_parser = subparsers.add_parser('baselines', help='Evaluate baseline policies')
    baselines_parser.add_argument('--episodes', type=int, default=10, help='Episodes per baseline')
    
    # Test env command
    test_parser = subparsers.add_parser('test-env', help='Test environment and config')
    
    # Plot rewards command
    plot_parser = subparsers.add_parser('plot-rewards', help='Plot rewards from training log')
    plot_parser.add_argument('--log', type=str, required=True, help='Path to reward_log.csv')
    plot_parser.add_argument('--output', type=str, help='Output image path (or show if not specified)')
    
    args = parser.parse_args()
    
    if args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'eval':
        cmd_eval(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'baselines':
        cmd_baselines(args)
    elif args.command == 'test-env':
        cmd_test_env(args)
    elif args.command == 'plot-rewards':
        cmd_plot_rewards(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
