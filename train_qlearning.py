"""Training script for tabular Q-Learning on CartPole-v1.

Produces reward CSVs, saves Q-table and generates plots.
"""

import os
import time
import numpy as np
import gymnasium as gym
from q_learning import QLearningAgent
from utils import set_seed, ensure_dir, save_rewards_csv
from plots import plot_rewards


def train(env_id: str = "CartPole-v1", episodes: int = 500, seed: int = 42, out_dir: str = "results/qlearning"):
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

    # generate plots
    plot_path = os.path.join(out_dir, f"learning_curve_{seed}_{timestamp}.png")
    plot_rewards(csv_path, plot_path, ma_window=20)

    print("Training complete. Outputs:")
    print(" - q-table:", q_path + ".npy")
    print(" - rewards CSV:", csv_path)
    print(" - plot:", plot_path)


if __name__ == "__main__":
    train()
