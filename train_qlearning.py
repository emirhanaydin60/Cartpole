"""Training script for tabular Q-Learning on CartPole-v1.

Produces reward CSVs, saves Q-table and generates plots.
"""

import os
import time
from typing import Sequence
import numpy as np
import gymnasium as gym
from q_learning import QLearningAgent
from utils import set_seed, ensure_dir, save_rewards_csv
from plots import plot_seed_comparison


def train_single_seed(env_id: str, episodes: int, seed: int, out_dir: str):
    set_seed(seed)
    ensure_dir(out_dir)

    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)

    agent = QLearningAgent(env)

    rewards = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state_idx = agent.discretize(obs)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.choose_action(state_idx)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_idx = agent.discretize(next_obs)
            agent.update(state_idx, action, reward, next_state_idx, done)
            state_idx = next_state_idx
            total_reward += reward
        agent.decay_epsilon()
        rewards.append(total_reward)
        if ep % 50 == 0:
            print(f"Episode {ep}/{episodes} reward={total_reward:.2f} eps={agent.epsilon:.3f}")

    timestamp = int(time.time())
    q_path = os.path.join(out_dir, f"q_table_{seed}_{timestamp}")
    agent.save(q_path)
    csv_path = os.path.join(out_dir, f"rewards_{seed}_{timestamp}.csv")
    save_rewards_csv(rewards, csv_path)

    print("Training complete. Outputs:")
    print(" - q-table:", q_path + ".npy")
    print(" - rewards CSV:", csv_path)
    return csv_path


def train(env_id: str = "CartPole-v1", episodes: int = 500, seeds: Sequence[int] = (42, 60, 100), out_dir: str = "results/qlearning"):
    ensure_dir(out_dir)

    csv_paths = []
    labels = []

    for index, seed in enumerate(seeds, start=1):
        print(f"Starting Q-Learning training for seed {seed} ({index}/{len(seeds)})")
        csv_path = train_single_seed(env_id, episodes, seed, out_dir)
        csv_paths.append(csv_path)
        labels.append(f"seed {index}")

    plot_path = os.path.join(out_dir, "qlearning_seed_comparison.png")
    plot_seed_comparison(csv_paths, labels, plot_path, title="Q-Learning Learning Curves by Seed", ma_window=20)
    print(" - combined plot:", plot_path)


if __name__ == "__main__":
    train()
