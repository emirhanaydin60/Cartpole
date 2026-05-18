"""Deep Q-Network (DQN) implementation using PyTorch.

Includes ReplayBuffer, Q-network, and agent utilities.
"""

from collections import deque, namedtuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from experiment_config import DEFAULT_DQN_CONFIG


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.transition = namedtuple("Transition", ["s", "a", "r", "s2", "done"])

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(self.transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s = np.vstack([t.s for t in batch])
        a = np.array([t.a for t in batch])
        r = np.array([t.r for t in batch], dtype=np.float32)
        s2 = np.vstack([t.s2 for t in batch])
        done = np.array([t.done for t in batch], dtype=np.uint8)
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, device="cpu", lr: float = DEFAULT_DQN_CONFIG.lr, gamma: float = DEFAULT_DQN_CONFIG.gamma, buffer_size: int = 10000, batch_size: int = DEFAULT_DQN_CONFIG.batch_size, target_update: int = 1000):
        self.device = torch.device(device)
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.update_counter = 0
        self.target_update = target_update

    def select_action(self, state: np.ndarray, epsilon: float):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q_net(state_t)
        return int(torch.argmax(qvals, dim=1).item())

    def store(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None
        s, a, r, s2, done = self.replay.sample(self.batch_size)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.tensor(s2, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(s_t).gather(1, a_t)
        with torch.no_grad():
            target_q = self.target_net(s2_t).max(1)[0].unsqueeze(1)
            target = r_t + (1.0 - done_t) * self.gamma * target_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # periodic target update
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)
