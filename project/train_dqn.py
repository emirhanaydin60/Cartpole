"""Training script for DQN on CartPole-v1.

Saves model .pth, reward CSV and plots.
"""
import os
import time
import random
import numpy as np
import gymnasium as gym
import torch
from .dqn import DQNAgent
from .utils import set_seed, ensure_dir, save_rewards_csv, save_model
from .plots import plot_rewards


def train(env_id: str = "CartPole-v1", episodes: int = 500, seed: int = 42, out_dir: str = "results/dqn",
          lr: float = 1e-3, gamma: float = 0.99, batch_size: int = 64):
    set_seed(seed)
    ensure_dir(out_dir)

    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

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
            total_reward += reward
        epsilon = max(eps_min, epsilon * eps_decay)
        rewards.append(total_reward)
        if ep % 50 == 0:
            print(f"Episode {ep}/{episodes} reward={total_reward:.2f} eps={epsilon:.3f}")

    timestamp = int(time.time())
    model_path = os.path.join(out_dir, f"dqn_model_{seed}_{timestamp}.pth")
    agent.save(model_path)
    csv_path = os.path.join(out_dir, f"rewards_{seed}_{timestamp}.csv")
    save_rewards_csv(rewards, csv_path)
    plot_path = os.path.join(out_dir, f"learning_curve_{seed}_{timestamp}.png")
    plot_rewards(csv_path, plot_path, ma_window=20)

    print("Training complete. Outputs:")
    print(" - model:", model_path)
    print(" - rewards CSV:", csv_path)
    print(" - plot:", plot_path)


if __name__ == "__main__":
    train()
