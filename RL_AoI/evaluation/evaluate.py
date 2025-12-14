#!/usr/bin/env python3
"""
Evaluation Utilities for AoI Minimization.

Provides tools to evaluate and compare trained RL agents against baselines.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from env.aoi_env import AoIEnv
from training.baselines import get_all_baselines, evaluate_baseline


def evaluate_model(
    model_path: str,
    config: Optional[Dict] = None,
    n_episodes: int = 20,
    deterministic: bool = True,
    verbose: bool = True,
    show_details: bool = True
) -> Dict:
    """
    Evaluate a trained PPO model.
    
    Args:
        model_path: Path to saved model
        config: Environment configuration
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic actions
        verbose: Whether to print progress
        show_details: Whether to show detailed step-by-step output
    
    Returns:
        Evaluation results dictionary
    """
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = AoIEnv(config=config)
    
    # Collect results
    episode_rewards = []
    episode_pose_aoi = []
    episode_style_aoi = []
    episode_total_aoi = []
    episode_pose_contrib = []
    episode_style_contrib = []
    action_distribution = np.zeros(10)
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        total_reward = 0
        pose_aoi_sum = 0
        style_aoi_sum = 0
        total_aoi_sum = 0
        pose_contrib_sum = 0
        style_contrib_sum = 0
        steps = 0
        
        if show_details and ep == 0:
            print("\n" + "=" * 100)
            print(f"Episode {ep + 1} - Detailed Step Output")
            print("=" * 100)
            header = f"{'Frame':<6} {'Decision':<22} {'Reward':<10} {'Pose Contrib':<14} {'Style Contrib':<14} {'Total AoI':<10} {'Obs?':<5}"
            print(header)
            print("-" * 100)
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_distribution[action] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            pose_aoi_sum += info['total_pose_aoi']
            style_aoi_sum += info['total_style_aoi']
            total_aoi_sum += info['total_aoi']
            pose_contrib_sum += info['pose_contribution']
            style_contrib_sum += info['style_contribution']
            steps += 1
            
            # Show detailed output for first episode
            if show_details and ep == 0 and steps <= 30:
                decision = info['decision']
                action_str = f"Cam{decision['camera_id']}-{decision['resolution_name']}"
                if not decision['action_taken']:
                    action_str += " (wait)"
                obs_str = "Yes" if info['observation_completed'] else ""
                
                print(f"{info['frame']:<6} {action_str:<22} {reward:<10.4f} "
                      f"{info['pose_contribution']:<14.4f} {info['style_contribution']:<14.4f} "
                      f"{info['total_aoi']:<10.4f} {obs_str:<5}")
            
            done = terminated or truncated
        
        if show_details and ep == 0:
            print("-" * 100)
            print(f"Episode Summary:")
            print(f"  Total Reward: {total_reward:.4f}")
            print(f"  Avg Pose Contribution: {pose_contrib_sum/steps:.4f} ({pose_contrib_sum/(pose_contrib_sum+style_contrib_sum)*100:.1f}%)")
            print(f"  Avg Style Contribution: {style_contrib_sum/steps:.4f} ({style_contrib_sum/(pose_contrib_sum+style_contrib_sum)*100:.1f}%)")
            print("=" * 100 + "\n")
        
        episode_rewards.append(total_reward)
        episode_pose_aoi.append(pose_aoi_sum / steps)
        episode_style_aoi.append(style_aoi_sum / steps)
        episode_total_aoi.append(total_aoi_sum / steps)
        episode_pose_contrib.append(pose_contrib_sum / steps)
        episode_style_contrib.append(style_contrib_sum / steps)
        
        if verbose:
            print(f"Episode {ep + 1}: reward={total_reward:.2f}, "
                  f"pose_contrib={pose_contrib_sum/steps:.4f}, "
                  f"style_contrib={style_contrib_sum/steps:.4f}, "
                  f"total_aoi={total_aoi_sum/steps:.4f}")
    
    # Normalize action distribution
    action_distribution /= action_distribution.sum()
    
    # Compute contribution percentages
    avg_pose_contrib = np.mean(episode_pose_contrib)
    avg_style_contrib = np.mean(episode_style_contrib)
    total_contrib = avg_pose_contrib + avg_style_contrib
    
    results = {
        'model_path': model_path,
        'n_episodes': n_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_pose_aoi': float(np.mean(episode_pose_aoi)),
        'mean_style_aoi': float(np.mean(episode_style_aoi)),
        'mean_total_aoi': float(np.mean(episode_total_aoi)),
        'mean_pose_contribution': float(avg_pose_contrib),
        'mean_style_contribution': float(avg_style_contrib),
        'pose_contribution_pct': float(avg_pose_contrib / total_contrib * 100) if total_contrib > 0 else 0,
        'style_contribution_pct': float(avg_style_contrib / total_contrib * 100) if total_contrib > 0 else 0,
        'action_distribution': action_distribution.tolist(),
        'episode_rewards': episode_rewards,
        'episode_total_aoi': episode_total_aoi
    }
    
    env.close()
    return results


def compare_with_baselines(
    model_path: str,
    config: Optional[Dict] = None,
    n_episodes: int = 20,
    save_path: Optional[str] = None
) -> Dict:
    """
    Compare trained model against all baselines.
    
    Args:
        model_path: Path to trained model
        config: Environment configuration
        n_episodes: Number of episodes per policy
        save_path: Path to save results (optional)
    
    Returns:
        Comparison results dictionary
    """
    print("=" * 60)
    print("Comparing PPO Agent vs Baselines")
    print("=" * 60)
    
    # Create environment
    env = AoIEnv(config=config)
    
    # Evaluate PPO model
    print("\n[1/2] Evaluating PPO Agent...")
    ppo_results = evaluate_model(model_path, config, n_episodes, verbose=False)
    print(f"  PPO Mean Reward: {ppo_results['mean_reward']:.2f} ± {ppo_results['std_reward']:.2f}")
    print(f"  PPO Mean AoI: {ppo_results['mean_total_aoi']:.4f}")
    
    # Evaluate baselines
    print("\n[2/2] Evaluating Baselines...")
    baselines = get_all_baselines()
    baseline_results = []
    
    for policy in baselines:
        result = evaluate_baseline(env, policy, n_episodes, verbose=False)
        baseline_results.append(result)
        print(f"  {policy.name}: reward={result['mean_reward']:.2f}, aoi={result['mean_total_aoi']:.4f}")
    
    env.close()
    
    # Compile results
    all_results = {
        'ppo': ppo_results,
        'baselines': baseline_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Sort all policies by reward
    all_policies = [('PPO', ppo_results['mean_reward'], ppo_results['mean_total_aoi'])]
    for br in baseline_results:
        all_policies.append((br['policy'], br['mean_reward'], br['mean_total_aoi']))
    
    all_policies.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Rank':<6}{'Policy':<30}{'Mean Reward':<15}{'Mean AoI':<15}")
    print("-" * 66)
    for i, (name, reward, aoi) in enumerate(all_policies):
        marker = " ***" if name == "PPO" else ""
        print(f"{i+1:<6}{name:<30}{reward:<15.2f}{aoi:<15.4f}{marker}")
    
    # Calculate improvement
    best_baseline_reward = max(br['mean_reward'] for br in baseline_results)
    improvement = ppo_results['mean_reward'] - best_baseline_reward
    improvement_pct = (improvement / abs(best_baseline_reward)) * 100 if best_baseline_reward != 0 else 0
    
    print(f"\nPPO vs Best Baseline:")
    print(f"  Reward improvement: {improvement:.2f} ({improvement_pct:+.1f}%)")
    
    # Save results
    if save_path is not None:
        with open(save_path, 'w') as f:
            # Convert numpy arrays for JSON serialization
            json_results = {
                'ppo': {k: v for k, v in ppo_results.items() if k != 'episode_rewards' and k != 'episode_total_aoi'},
                'baselines': [{k: v for k, v in br.items() if k != 'episode_rewards'} for br in baseline_results],
                'ranking': [(name, float(reward), float(aoi)) for name, reward, aoi in all_policies],
                'improvement': {
                    'absolute': float(improvement),
                    'percentage': float(improvement_pct)
                },
                'timestamp': all_results['timestamp']
            }
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {save_path}")
    
    return all_results


def plot_training_curves(log_dir: str, save_path: Optional[str] = None):
    """
    Plot training curves from tensorboard logs.
    
    Args:
        log_dir: Directory containing tensorboard logs
        save_path: Path to save figure (optional)
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed. Skipping plot.")
        return
    
    # Find event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"No tensorboard events found in {log_dir}")
        return
    
    # Load events
    ea = EventAccumulator(event_files[0])
    ea.Reload()
    
    # Get available scalars
    scalar_tags = ea.Tags()['scalars']
    print(f"Available metrics: {scalar_tags}")
    
    # Plot reward curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = [
        ('rollout/ep_rew_mean', 'Episode Reward'),
        ('rollout/ep_len_mean', 'Episode Length'),
        ('train/value_loss', 'Value Loss'),
        ('train/policy_gradient_loss', 'Policy Loss')
    ]
    
    for ax, (tag, title) in zip(axes.flat, metrics):
        if tag in scalar_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values)
            ax.set_title(title)
            ax.set_xlabel('Steps')
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{title} (not found)")
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_action_distribution(results: Dict, save_path: Optional[str] = None):
    """
    Plot action distribution of the trained agent.
    
    Args:
        results: Evaluation results from evaluate_model
        save_path: Path to save figure (optional)
    """
    action_dist = np.array(results['action_distribution'])
    
    # Labels for actions
    labels = [
        'Cam0-Res1', 'Cam1-Res1', 'Cam2-Res1', 'Cam3-Res1', 'Cam4-Res1',
        'Cam0-Res2', 'Cam1-Res2', 'Cam2-Res2', 'Cam3-Res2', 'Cam4-Res2'
    ]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = ['#3498db'] * 5 + ['#e74c3c'] * 5  # Blue for Res1, Red for Res2
    bars = ax.bar(range(10), action_dist * 100, color=colors)
    
    ax.set_xticks(range(10))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Action Frequency (%)')
    ax.set_title('PPO Agent Action Distribution')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Resolution 1 (640x360)'),
        Patch(facecolor='#e74c3c', label='Resolution 2 (1280x720)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_aoi_trajectory(
    model_path: str,
    config: Optional[Dict] = None,
    save_path: Optional[str] = None
):
    """
    Plot AoI trajectory over one episode.
    
    Args:
        model_path: Path to trained model
        config: Environment configuration
        save_path: Path to save figure (optional)
    """
    # Load model
    model = PPO.load(model_path)
    env = AoIEnv(config=config)
    
    # Run one episode
    obs, info = env.reset()
    
    pose_aoi_history = []
    style_aoi_history = []
    total_aoi_history = []
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        pose_aoi_history.append(info['pose_aoi'])
        style_aoi_history.append(info['style_aoi'])
        total_aoi_history.append(info['total_aoi'])
        
        done = terminated or truncated
    
    env.close()
    
    # Convert to arrays
    pose_aoi = np.array(pose_aoi_history)
    style_aoi = np.array(style_aoi_history)
    total_aoi = np.array(total_aoi_history)
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Pose AoI per object
    ax1 = axes[0]
    for i in range(5):
        ax1.plot(pose_aoi[:, i], label=f'Object {i}')
    ax1.set_ylabel('Pose AoI')
    ax1.set_title('Pose AoI Trajectory')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Style AoI per object
    ax2 = axes[1]
    for i in range(5):
        ax2.plot(style_aoi[:, i], label=f'Object {i}')
    ax2.set_ylabel('Style AoI')
    ax2.set_title('Style AoI Trajectory')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Total AoI
    ax3 = axes[2]
    ax3.plot(total_aoi, 'k-', linewidth=2)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Total AoI')
    ax3.set_title('Total Weighted AoI Trajectory')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate AoI RL Model")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare with baselines"
    )
    parser.add_argument(
        "--plot-trajectory", action="store_true",
        help="Plot AoI trajectory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./evaluation_results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.compare:
        results = compare_with_baselines(
            args.model,
            n_episodes=args.episodes,
            save_path=os.path.join(args.output_dir, "comparison.json")
        )
        
        # Plot action distribution
        if 'action_distribution' in results.get('ppo', {}):
            plot_action_distribution(
                results['ppo'],
                save_path=os.path.join(args.output_dir, "action_distribution.png")
            )
    else:
        results = evaluate_model(args.model, n_episodes=args.episodes)
        print(f"\nResults: {results}")
    
    if args.plot_trajectory:
        plot_aoi_trajectory(
            args.model,
            save_path=os.path.join(args.output_dir, "aoi_trajectory.png")
        )


if __name__ == "__main__":
    main()
