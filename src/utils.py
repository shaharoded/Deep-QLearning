"""
Utility classes and functions for DQN agents.
"""

import numpy as np
import torch
import random
from collections import deque, namedtuple
from typing import Tuple


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience replay buffer for DQN agents.
    Stores transitions and samples random minibatches for training.
    
    This buffer is updated as new experiences are collected.

    When? After each step, if the buffer has enough samples (≥ batch_size), 
    we sample a random batch and perform one gradient update

    NOTE: This is a simple uniform replay buffer. A possible improvement will implement prioritized experience replay.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions based on their TD error (priority).
    Transitions with higher TD error are sampled more frequently.
    
    Reference: Schaul et al. (2015) - Prioritized Experience Replay
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Amount to increment beta per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        
        # Storage
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, td_error: float = None):
        """
        Add a transition to the buffer with priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            td_error: TD error (priority). If None, uses max priority.
        """
        # Use max priority for new experiences
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        else:
            priority = max_priority
        
        # Store experience
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, 
                                                 torch.Tensor, torch.Tensor, 
                                                 torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Sample a batch of transitions based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
            weights: Importance sampling weights to correct bias
            indices: Indices of sampled experiences (for priority updates)
        """
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Increment beta
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor([e.done for e in experiences])
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on new TD errors.
        
        Args:
            indices: Indices of experiences to update
            td_errors: New TD errors
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size