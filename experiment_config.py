"""Centralized experiment settings for Q-Learning and DQN runs.

Edit the values in this file to change the default training setup in one place.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 500
    alpha: float = 0.001
    gamma: float = 0.8
    epsilon: float = 0.8
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9
    n_bins: tuple[int, int, int, int] = (6, 12, 6, 12)
    seeds: tuple[int, int, int] = (42, 60, 100)
    out_dir: str = "results/qlearning"


@dataclass(frozen=True)
class DQNConfig:
    episodes: int = 500
    lr: float = 1e-4
    gamma: float = 0.99
    epsilon: float = 0.95
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    seeds: tuple[int, int, int] = (42, 60, 100)
    out_dir: str = "results/dqn"


DEFAULT_QLEARNING_CONFIG = QLearningConfig()
DEFAULT_DQN_CONFIG = DQNConfig()


def format_qlearning_box(config: QLearningConfig = DEFAULT_QLEARNING_CONFIG) -> str:
    return "\n".join([f"alpha/lr={config.alpha}", f"gamma/dis.fac.={config.gamma}", f"epsilon/exp.rate.={config.epsilon}"])


def format_dqn_box(config: DQNConfig = DEFAULT_DQN_CONFIG) -> str:
    return "\n".join([f"alpha/lr={config.lr}", f"gamma/dis.fac.={config.gamma}", f"epsilon/exp.rate.={config.epsilon}"])
