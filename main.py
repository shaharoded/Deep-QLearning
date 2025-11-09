"""
Main training and evaluation script for Deep RL Assignment.

This script provides examples and utilities for training and evaluating
the different agent implementations.
"""

import gymnasium as gym
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
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
                num_episodes: int = 500, config: Dict = None,
                max_steps: int = 1000, # Add max_steps
                save_q_table_at: Optional[List[int]] = None,
                env_kwargs: Optional[Dict] = None):
    """
    Train a specific agent on an environment.
    
Args:
        agent_class: Agent class to instantiate
        env_name: Gymnasium environment name
        num_episodes: Number of training episodes
        config: Configuration dictionary for the agent
        max_steps: Max steps per episode
        save_q_table_at: List of episodes to log Q-table
        env_kwargs: Arguments for gym.make()
        
    Returns:
        Tuple of (trained_agent, training_metrics)
    """
    # Create environment
    if env_kwargs is None:
        env_kwargs = {}
    env = gym.make(env_name, **env_kwargs)
    
    # Get dimensions
    if isinstance(env.observation_space, gym.spaces.Box):
        state_dim = env.observation_space.shape[0]
    else: # Handles Discrete spaces like FrozenLake
        state_dim = env.observation_space.n
    
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = agent_class(state_dim=state_dim, action_dim=action_dim, config=config)
    
    print(f"\n{'='*60}")
    print(f"Training {agent.__class__.__name__} on {env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Episodes: {num_episodes}, Max steps/ep: {max_steps}")
    print(f"{'='*60}\n")
    
    # Train
    # Pass max_steps and save_q_table_at
    metrics = agent.train(env, num_episodes=num_episodes, max_steps=max_steps, 
                          verbose=True, save_q_table_at=save_q_table_at)
    
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

def plot_section1_results(metrics: Dict[str, Any], agent_name: str, save_path: str = None):
    """
    Plot training metrics for Section 1 (Rewards and Steps).
    
    Args:
        metrics: Dictionary containing training metrics
        agent_name: Name of the agent for the title
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    window = 100
    
    # --- Plot 1: Rewards ---
    rewards = metrics['rewards']
    axes[0].plot(rewards, alpha=0.6, label='Episode Reward')
    
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), moving_avg, 
                    label=f'{window}-Episode Moving Average', linewidth=2, color='C1')
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title(f'{agent_name} - Training Rewards per Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # --- Plot 2: Steps to Goal ---
    lengths = metrics['lengths']
    axes[1].plot(lengths, alpha=0.6, label='Episode Length')

    if len(lengths) >= window:
        moving_avg_steps = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(lengths)), moving_avg_steps, 
                     label=f'{window}-Episode Moving Average', linewidth=2, color='C2')
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].set_title(f'{agent_name} - Steps to Goal per Episode')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_q_table_heatmap(q_table: np.ndarray, title: str, save_path: str = None):
    """
    Plot the Q-table as a heatmap of V(s) = max_a Q(s,a).
    
    Args:
        q_table: The Q-table (state_dim, action_dim)
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    if q_table is None or q_table.size == 0:
        print(f"Skipping plot: {title} (Q-table is empty)")
        return
        
    # FrozenLake is 4x4, so 16 states
    try:
        grid_size = int(np.sqrt(q_table.shape[0]))
        if grid_size * grid_size != q_table.shape[0]:
            print(f"Cannot plot Q-table: Not a square grid.")
            return
    except:
        print(f"Cannot plot Q-table: Error determining grid size.")
        return

    # Get V(s) = max_a Q(s,a)
    v_values = np.max(q_table, axis=1)
    v_grid = v_values.reshape((grid_size, grid_size))
    
    # Get policy (pi(s) = argmax_a Q(s,a))
    policy = np.argmax(q_table, axis=1)
    policy_grid = policy.reshape((grid_size, grid_size))
    
    # Actions: 0:Left, 1:Down, 2:Right, 3:Up
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    # Create text labels: "V(s)\npi(s)"
    labels = np.full_like(v_grid, "", dtype=object)
    for r in range(grid_size):
        for c in range(grid_size):
            v = v_grid[r, c]
            p = policy_grid[r, c]
            labels[r, c] = f"{v:.2f}\n{action_symbols[p]}"

    # Plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(v_grid, annot=labels, fmt="", cmap="viridis",
                linewidths=0.5, linecolor='black')
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        
    plt.show()

def run_section_1_frozenlake():
    """Run the Q-Learning experiment on FrozenLake-v0."""
    print("\n" + "="*60)
    print("SECTION 1: Tabular Q-Learning on FrozenLake-v1")
    print("="*60)
    
    env_name = 'FrozenLake-v1'
    num_episodes = 5000 # As per hw1.docx
    max_steps = 100    # As per hw1.docx
    save_q_at = [500, 2000, num_episodes] # As per hw1.docx
    
    # Hyperparameters to optimize
    config = {
        'alpha': 0.1,             # Learning rate
        'gamma': 0.99,            # Discount factor
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.9995 # Slower decay for more episodes
    }
    
    # Run training
    agent, metrics = train_agent(
        QLearningAgent,
        env_name=env_name,
        num_episodes=num_episodes,
        max_steps=max_steps,
        config=config,
        save_q_table_at=save_q_at,
        env_kwargs={'is_slippery': True} # Standard environment
    )
    
    # --- Plotting Results ---
    
    # 1. Plot reward per episode and steps to goal
    plot_section1_results(metrics, "Q-Learning (FrozenLake)", 
                          save_path='results/section1/frozenlake_training.png')
    
    # 2. Plot Q-value tables
    q_tables = metrics.get('q_tables', {})
    for episode, q_table in q_tables.items():
        title = f"Q-Table (V(s) and Policy) at Episode {episode}"
        plot_q_table_heatmap(q_table, title, 
                             save_path=f'results/section1/frozenlake_q_table_ep{episode}.png')
    
    return agent, metrics

def main():
    """Main function with different execution modes."""
    print("\n" + "="*60)
    print("DEEP REINFORCEMENT LEARNING - ASSIGNMENT 1")
    print("="*60)
    print("\nAvailable modes:")
    print("0. Run Section 1 (Q-Learning on FrozenLake)") # Added
    print("1. Run CartPole example (Improved DQN)")
    print("2. Compare agents on CartPole")
    print("3. Custom environment (modify run_example_custom_env)")
    print("="*60)
    
    # Choose mode
    mode = 0  # Change this to 0 to run the Section 1 code
    
    if mode == 0:
        # Run Section 1
        agent, metrics = run_section_1_frozenlake()

    elif mode == 1:
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
