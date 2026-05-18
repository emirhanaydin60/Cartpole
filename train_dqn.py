"""Training script for DQN on CartPole-v1.

Saves model .pth, reward CSV and plots.
"""

import os
import time
from typing import Sequence
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
import torch
from dqn import DQNAgent
from utils import set_seed, ensure_dir, save_rewards_csv
from plots import plot_seed_comparison


def train_single_seed(env_id: str, episodes: int, seed: int, out_dir: str, lr: float, gamma: float, batch_size: int):
    set_seed(seed)
    ensure_dir(out_dir)

    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    if env.observation_space.shape is None:
        raise ValueError("Expected a vector observation space with a fixed shape.")
    state_dim = int(env.observation_space.shape[0])
    if not isinstance(env.action_space, Discrete):
        raise ValueError("Expected a discrete action space for CartPole-v1.")
    action_dim = int(env.action_space.n)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(state_dim, action_dim, device=device, lr=lr, gamma=gamma, batch_size=batch_size)

    epsilon = 1.0
    eps_min = 0.01
    eps_decay = 0.995

    rewards = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            obs = next_obs
            total_reward += float(reward)
        epsilon = max(eps_min, epsilon * eps_decay)
        rewards.append(total_reward)
        if ep % 25 == 0:
            print(f"Episode {ep}/{episodes} reward={total_reward:.2f} eps={epsilon:.3f}")

    timestamp = int(time.time())
    model_path = os.path.join(out_dir, f"dqn_model_{seed}_{timestamp}.pth")
    agent.save(model_path)
    csv_path = os.path.join(out_dir, f"rewards_{seed}_{timestamp}.csv")
    save_rewards_csv(rewards, csv_path)

    print("Training complete. Outputs:")
    print(" - model:", model_path)
    print(" - rewards CSV:", csv_path)
    return csv_path


def train(env_id: str = "CartPole-v1", episodes: int = 500, seeds: Sequence[int] = (42, 60, 100), out_dir: str = "results/dqn", lr: float = 1e-3, gamma: float = 0.99, batch_size: int = 64):
    ensure_dir(out_dir)
    run_id = int(time.time())

    csv_paths = []
    labels = []
    annotation_texts = []

    for index, seed in enumerate(seeds, start=1):
        print(f"Starting DQN training for seed {seed} ({index}/{len(seeds)})")
        csv_path = train_single_seed(env_id, episodes, seed, out_dir, lr, gamma, batch_size)
        csv_paths.append(csv_path)
        labels.append(str(index))
        annotation_texts.append(
            f"seed={seed}\n"
            f"episodes={episodes}\n"
            f"lr={lr}\n"
            f"gamma={gamma}\n"
            f"batch={batch_size}"
        )

    plot_path = os.path.join(out_dir, f"dqn_seed_comparison_{run_id}.png")
    plot_seed_comparison(
        csv_paths,
        labels,
        plot_path,
        title="DQN Learning Curves by Seed",
        ma_window=20,
        annotation_texts=annotation_texts,
    )
    print(" - combined plot:", plot_path)


if __name__ == "__main__":
    train()
