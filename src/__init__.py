"""
Deep Reinforcement Learning Assignment - Source Module
"""

from .agent import (
    Agent,
    QLearningAgent,
    DeepQLearningAgent,
    DoubleDeepQLearningAgent
)

__all__ = [
    'Agent',
    'QLearningAgent',
    'DeepQLearningAgent',
    'DoubleDeepQLearningAgent'
]

__version__ = '1.0.0'
