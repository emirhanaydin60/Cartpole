"""Tabular Q-Learning implementation with state discretization for CartPole-v1.

This module provides a QLearningAgent class that discretizes continuous states
into bins and performs tabular Q-Learning with epsilon-greedy exploration.
"""

from typing import List, Tuple
import numpy as np
import gymnasium as gym
from utils import save_qtable


class QLearningAgent:
    def __init__(self, env: gym.Env, n_bins: List[int] = None, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995):
        """Initialize the tabular Q-Learning agent.

        Args:
            env: Gym environment (CartPole-v1 expected).
            n_bins: Number of discrete bins for each state dimension (list of 4 ints).
            alpha: Learning rate.
            gamma: Discount factor.
            epsilon: Initial exploration probability.
            epsilon_min: Minimum epsilon value.
            epsilon_decay: Multiplicative decay per episode.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # default bins if not provided
        if n_bins is None:
            n_bins = [6, 12, 6, 12]
        self.n_bins = n_bins

        # define observation bounds (CartPole-v1)
        self.obs_space_low = np.array([-4.8, -5.0, -0.418, -5.0])
        self.obs_space_high = np.array([4.8, 5.0, 0.418, 5.0])

        # create bins for each dimension
        self.bins = [np.linspace(self.obs_space_low[i], self.obs_space_high[i], self.n_bins[i] - 1) for i in range(4)]

        # Q-table shape: bins per dimension + actions
        action_size = env.action_space.n
        self.q_table = np.zeros(tuple(self.n_bins) + (action_size,))

    def discretize(self, obs: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert continuous observation into discrete bin indices."""
        discretized = []
        for i, val in enumerate(obs):
            idx = np.digitize(val, self.bins[i])
            if idx < 0:
                idx = 0
            elif idx >= self.n_bins[i]:
                idx = self.n_bins[i] - 1
            discretized.append(int(idx))
        return tuple(discretized)

    def choose_action(self, state_idx: Tuple[int, int, int, int]) -> int:
        """Epsilon-greedy action selection on discretized state."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        q_vals = self.q_table[state_idx]
        return int(np.argmax(q_vals))

    def update(self, state_idx, action: int, reward: float, next_state_idx, done: bool):
        """Q-Learning update rule.

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        """
        current = self.q_table[state_idx + (action,)]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])
        self.q_table[state_idx + (action,)] = current + self.alpha * (target - current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        save_qtable(self.q_table, path)
