"""Plotting utilities for learning curves and comparisons.

All plots are saved as publication-quality PNG files.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def moving_average(x, window=20):
    return np.convolve(x, np.ones(window) / window, mode='valid')


def plot_rewards(csv_path: str, out_path: str, ma_window: int = 20, title: str = None):
    df = pd.read_csv(csv_path)
    episodes = df['episode'].values
    rewards = df['reward'].values

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, rewards, label='Episode reward', alpha=0.6)
    if len(rewards) >= ma_window:
        ma = moving_average(rewards, window=ma_window)
        plt.plot(episodes[ma_window - 1:], ma, label=f'{ma_window}-episode MA', color='tab:red')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title or 'Learning Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def compare_reward_files(csv_paths: list, labels: list, out_path: str, ma_window: int = 20, title: str = None):
    plt.figure(figsize=(8, 5))
    for p, lab in zip(csv_paths, labels):
        df = pd.read_csv(p)
        rewards = df['reward'].values
        episodes = df['episode'].values
        plt.plot(episodes, rewards, alpha=0.4)
        if len(rewards) >= ma_window:
            ma = moving_average(rewards, window=ma_window)
            plt.plot(episodes[ma_window - 1:], ma, label=lab)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title or 'Comparison')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
