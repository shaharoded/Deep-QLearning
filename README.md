# Deep Reinforcement Learning - Assignment 1

## Project Overview
This project implements and compares three different Q-Learning approaches for reinforcement learning tasks:
- Traditional Q-Learning
- Deep Q-Learning (DQN)
- Improved Deep Q-Learning (with enhancements)

## Requirements
- Python 3.8+
- Gymnasium (OpenAI Gym)
- PyTorch
- NumPy
- Matplotlib (for visualization)

## Installation

### Quick Setup
```bash
# Install all dependencies
pip install -r requirements.txt
```

### Manual Installation
```bash
pip install gymnasium torch numpy matplotlib
```

## Project Structure
```
Assignment1/
├── agent.py              # Agent base class and implementations
│   ├── Agent             # Base class with common interface
│   ├── QLearningAgent    # Tabular Q-Learning
│   ├── DeepQLearningAgent        # DQN implementation
│   └── ImprovedDeepQLearningAgent # Enhanced DQN
├── main.py               # Main training and evaluation scripts
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── results/             # Training plots and logs
└── models/              # Saved model checkpoints
```

## Quick Start

### Run the Example
The easiest way to get started:

```bash
python main.py
```

This will train an Improved DQN agent on CartPole-v1 and save results to `results/` and models to `models/`.

### Training an Agent (Custom Code)
```python
from agent import DeepQLearningAgent, ImprovedDeepQLearningAgent
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')

# Configure agent
config = {
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

# Initialize agent
agent = ImprovedDeepQLearningAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    config=config
)

# Train agent
metrics = agent.train(env, num_episodes=500)

# Evaluate
mean_reward, std_reward = agent.evaluate(env, num_episodes=20)
print(f"Performance: {mean_reward:.2f} ± {std_reward:.2f}")

# Save model
agent.save('models/my_agent.pth')
```

## Agents

### Common API
All agents inherit from the `Agent` base class and share a common interface:

**Core Methods:**
- `select_action(state, training=True)` - Choose an action (with exploration if training)
- `update(state, action, reward, next_state, done)` - Learn from a transition
- `train(env, num_episodes, ...)` - Complete training loop
- `evaluate(env, num_episodes=10)` - Evaluate performance
- `save(filepath)` / `load(filepath)` - Persist agent state

**Configuration:**
All agents accept a `config` dictionary with hyperparameters like:
- `learning_rate`: Learning rate for updates
- `gamma`: Discount factor for future rewards
- `epsilon_start/min/decay`: Exploration parameters

### 1. QLearning (Tabular Q-Learning)
Traditional Q-Learning with tabular representation. Suitable for discrete state spaces.

**Key Features:**
- Q-table for state-action values
- ε-greedy exploration
- Temporal Difference learning

### 2. DeepQLearning (DQN)
Deep Q-Network using neural networks to approximate Q-values.

**Key Features:**
- Neural network for function approximation
- Experience replay buffer
- Target network for stable learning
- ε-greedy exploration with decay

### 3. ImprovedDeepQLearning
Enhanced DQN with modern improvements.

**Key Features:**
- Double DQN
- Dueling architecture
- Prioritized experience replay
- Noisy networks for exploration

## Evaluation
Each agent can be evaluated on:
- Training performance (reward over episodes)
- Sample efficiency
- Final policy performance
- Convergence stability

## Results
Results including training curves and performance metrics will be saved in the `results/` directory.

## References
- Sutton & Barto - Reinforcement Learning: An Introduction
- Mnih et al. - Playing Atari with Deep Reinforcement Learning (DQN)
- Van Hasselt et al. - Deep Reinforcement Learning with Double Q-learning
- Wang et al. - Dueling Network Architectures for Deep Reinforcement Learning
