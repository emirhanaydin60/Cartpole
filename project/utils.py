import os
import random
import json
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int):
    """Set random seeds for reproducibility across numpy, random and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_rewards_csv(rewards, path: str):
    df = pd.DataFrame({"episode": np.arange(1, len(rewards) + 1), "reward": rewards})
    df.to_csv(path, index=False)


def load_rewards_csv(path: str):
    return pd.read_csv(path)


def save_qtable(q_table: np.ndarray, path: str):
    np.save(path, q_table)


def load_qtable(path: str):
    return np.load(path + ".npy")


def save_model(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
