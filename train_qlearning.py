"""Training script for tabular Q-Learning on CartPole-v1.

Produces reward CSVs, saves Q-table and generates plots.
"""

import os
import time
import numpy as np
import gymnasium as gym
from q_learning import QLearningAgent
from utils import set_seed, ensure_dir, save_rewards_csv
from plots import plot_seed_comparison
from experiment_config import DEFAULT_QLEARNING_CONFIG, format_qlearning_box


def _next_comparison_index(out_dir: str, prefix: str) -> int:
    existing_indices = []
    for filename in os.listdir(out_dir):
        if filename.startswith(prefix) and filename.endswith(".png"):
            middle = filename[len(prefix) + 1 : -4]
            if middle.isdigit():
                existing_indices.append(int(middle))
    return max(existing_indices, default=0) + 1


def train_single_seed(env_id: str, config, seed: int, out_dir: str):
    set_seed(seed)
    ensure_dir(out_dir)

    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)

    agent = QLearningAgent(
        env,
        n_bins=list(config.n_bins),
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
    )

    rewards = []

    for ep in range(1, config.episodes + 1):
        obs, _ = env.reset()
        state_idx = agent.discretize(obs)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.choose_action(state_idx)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_idx = agent.discretize(next_obs)
            reward_value = float(reward)
            agent.update(state_idx, action, reward_value, next_state_idx, done)
            state_idx = next_state_idx
            total_reward += reward_value
        agent.decay_epsilon()
        rewards.append(total_reward)
        if ep % 50 == 0:
            print(f"Episode {ep}/{config.episodes} reward={total_reward:.2f} eps={agent.epsilon:.3f}")

    timestamp = int(time.time())
    q_path = os.path.join(out_dir, f"q_table_{seed}_{timestamp}")
    agent.save(q_path)
    csv_path = os.path.join(out_dir, f"rewards_{seed}_{timestamp}.csv")
    save_rewards_csv(rewards, csv_path)

    print("Training complete. Outputs:")
    print(" - q-table:", q_path + ".npy")
    print(" - rewards CSV:", csv_path)
    return csv_path


def train(env_id: str = "CartPole-v1", config=DEFAULT_QLEARNING_CONFIG):
    ensure_dir(config.out_dir)
    comparison_index = _next_comparison_index(config.out_dir, "qlearning_seed_comp")

    csv_paths = []
    labels = []
    figure_annotation_text = format_qlearning_box(config)

    for index, seed in enumerate(config.seeds, start=1):
        print(f"Starting Q-Learning training for seed {seed} ({index}/{len(config.seeds)})")
        csv_path = train_single_seed(env_id, config, seed, config.out_dir)
        csv_paths.append(csv_path)
        labels.append(f"Seed {index}")

    plot_path = os.path.join(config.out_dir, f"qlearning_seed_comp_{comparison_index}.png")
    plot_seed_comparison(
        csv_paths,
        labels,
        plot_path,
        title="Q-Learning Learning Curves by Seed",
        ma_window=20,
        figure_annotation_text=figure_annotation_text,
    )
    print(" - combined plot:", plot_path)


if __name__ == "__main__":
    train()
