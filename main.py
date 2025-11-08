"""
Main training and evaluation script for Deep RL Assignment.

This script provides examples and utilities for training and evaluating
the different agent implementations.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import os

from agent import QLearningAgent, DeepQLearningAgent, ImprovedDeepQLearningAgent


def plot_training_results(metrics: Dict[str, List[float]], agent_name: str, save_path: str = None):
    """
    Plot training metrics.
    
    Args:
        metrics: Dictionary containing training metrics
        agent_name: Name of the agent for the title
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    axes[0].plot(metrics['rewards'], alpha=0.6, label='Episode Reward')
    
    # Compute moving average
    window = 100
    if len(metrics['rewards']) >= window:
        moving_avg = np.convolve(metrics['rewards'], 
                                 np.ones(window)/window, 
                                 mode='valid')
        axes[0].plot(range(window-1, len(metrics['rewards'])), 
                    moving_avg, 
                    label=f'{window}-Episode Moving Average',
                    linewidth=2)
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title(f'{agent_name} - Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot losses if available
    if metrics['losses']:
        axes[1].plot(metrics['losses'], alpha=0.7, color='red')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'{agent_name} - Training Loss')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def train_agent(agent_class, env_name: str = 'CartPole-v1', 
                num_episodes: int = 500, config: Dict = None):
    """
    Train a specific agent on an environment.
    
    Args:
        agent_class: Agent class to instantiate
        env_name: Gymnasium environment name
        num_episodes: Number of training episodes
        config: Configuration dictionary for the agent
        
    Returns:
        Tuple of (trained_agent, training_metrics)
    """
    # Create environment
    env = gym.make(env_name)
    
    # Get dimensions
    if isinstance(env.observation_space, gym.spaces.Box):
        state_dim = env.observation_space.shape[0]
    else:
        state_dim = env.observation_space.n
    
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = agent_class(state_dim=state_dim, action_dim=action_dim, config=config)
    
    print(f"\n{'='*60}")
    print(f"Training {agent.__class__.__name__} on {env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*60}\n")
    
    # Train
    metrics = agent.train(env, num_episodes=num_episodes, verbose=True)
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating trained agent...")
    mean_reward, std_reward = agent.evaluate(env, num_episodes=20)
    print(f"Evaluation: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")
    print("="*60 + "\n")
    
    env.close()
    
    return agent, metrics


def compare_agents(env_name: str = 'CartPole-v1', num_episodes: int = 500):
    """
    Compare all three agent types on the same environment.
    
    Args:
        env_name: Gymnasium environment name
        num_episodes: Number of training episodes per agent
    """
    print("\n" + "="*60)
    print("COMPARING ALL AGENTS")
    print("="*60)
    
    results = {}
    
    # DQN configuration
    dqn_config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'buffer_capacity': 10000,
        'target_update_freq': 10,
        'hidden_dims': [128, 128]
    }
    
    # Improved DQN configuration
    improved_config = dqn_config.copy()
    improved_config.update({
        'use_double_dqn': True,
        'use_dueling': True,
        'use_prioritized_replay': False
    })
    
    # Train Deep Q-Learning agent
    print("\n[1/2] Training DeepQLearningAgent...")
    dqn_agent, dqn_metrics = train_agent(
        DeepQLearningAgent, 
        env_name=env_name,
        num_episodes=num_episodes,
        config=dqn_config
    )
    results['DQN'] = {
        'agent': dqn_agent,
        'metrics': dqn_metrics
    }
    
    # Train Improved Deep Q-Learning agent
    print("\n[2/2] Training ImprovedDeepQLearningAgent...")
    improved_agent, improved_metrics = train_agent(
        ImprovedDeepQLearningAgent,
        env_name=env_name,
        num_episodes=num_episodes,
        config=improved_config
    )
    results['Improved DQN'] = {
        'agent': improved_agent,
        'metrics': improved_metrics
    }
    
    # Note: Q-Learning (tabular) is not suitable for CartPole's continuous state space
    # It would require discretization, which is left as an exercise
    
    # Plot comparison
    plot_comparison(results, env_name)
    
    return results


def plot_comparison(results: Dict, env_name: str, save_path: str = None):
    """
    Plot comparison of multiple agents.
    
    Args:
        results: Dictionary of agent results
        env_name: Environment name for the title
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    window = 50
    
    for name, data in results.items():
        rewards = data['metrics']['rewards']
        
        # Plot raw rewards (transparent)
        ax.plot(rewards, alpha=0.2)
        
        # Plot moving average
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 
                   label=name, linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title(f'Agent Comparison on {env_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def run_example_cartpole():
    """Run a complete example on CartPole environment."""
    print("\n" + "="*60)
    print("CARTPOLE EXAMPLE")
    print("="*60)
    
    # Configuration
    env_name = 'CartPole-v1'
    num_episodes = 500
    
    # Train Improved DQN
    config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'buffer_capacity': 10000,
        'target_update_freq': 10,
        'hidden_dims': [128, 128],
        'use_double_dqn': True,
        'use_dueling': True
    }
    
    agent, metrics = train_agent(
        ImprovedDeepQLearningAgent,
        env_name=env_name,
        num_episodes=num_episodes,
        config=config
    )
    
    # Plot results
    plot_training_results(metrics, 'Improved DQN', 
                         save_path='results/cartpole_training.png')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    agent.save('models/improved_dqn_cartpole.pth')
    
    return agent, metrics


def run_example_custom_env():
    """
    Template for running on a custom environment.
    Modify this function for different Gymnasium environments.
    """
    # Example: LunarLander-v2, MountainCar-v0, Acrobot-v1, etc.
    env_name = 'LunarLander-v2'  # Change this
    
    # Adjust hyperparameters based on environment
    config = {
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 128,
        'buffer_capacity': 50000,
        'target_update_freq': 100,
        'hidden_dims': [256, 256],
        'use_double_dqn': True,
        'use_dueling': True
    }
    
    print(f"\nTraining on {env_name}")
    print("Note: Install required dependencies if environment is not available")
    
    try:
        agent, metrics = train_agent(
            ImprovedDeepQLearningAgent,
            env_name=env_name,
            num_episodes=1000,
            config=config
        )
        
        plot_training_results(metrics, f'Improved DQN - {env_name}')
        
        return agent, metrics
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the environment is installed and available.")
        return None, None


def main():
    """Main function with different execution modes."""
    print("\n" + "="*60)
    print("DEEP REINFORCEMENT LEARNING - ASSIGNMENT 1")
    print("="*60)
    print("\nAvailable modes:")
    print("1. Run CartPole example (Improved DQN)")
    print("2. Compare agents on CartPole")
    print("3. Custom environment (modify run_example_custom_env)")
    print("="*60)
    
    # Choose mode
    mode = 1  # Change this to select different modes
    
    if mode == 1:
        # Run single agent example
        agent, metrics = run_example_cartpole()
        
    elif mode == 2:
        # Compare agents
        results = compare_agents(env_name='CartPole-v1', num_episodes=500)
        
    elif mode == 3:
        # Custom environment
        agent, metrics = run_example_custom_env()
    
    print("\n" + "="*60)
    print("Execution completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run main
    main()
