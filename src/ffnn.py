"""
Feed-forward neural network architectures for Q-value approximation.
"""

import torch
import torch.nn as nn
from typing import List


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    
    Simple feedforward network with configurable architecture.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        """
        Initialize Q-network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: List of hidden layer dimensions (e.g., [64, 64, 64] for 3 layers)
        """
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Store architecture info
        self.hidden_dims = hidden_dims
        self.num_hidden_layers = len(hidden_dims)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,)
            
        Returns:
            Q-values for all actions (batch_size, action_dim) or (action_dim,)
        """
        return self.network(state)