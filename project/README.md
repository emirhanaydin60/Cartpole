# Theoretical and Experimental Comparison of Reinforcement Learning Algorithms

Project comparing tabular Q-Learning and Deep Q-Network (DQN) on Gymnasium `CartPole-v1`.

Prerequisites
- Create and activate conda env: `conda activate RL_v1` (Important: her kodu çalıştırmadan önce conda activate RL_v1 demeyi unutma.)
- Install dependencies: `pip install -r requirements.txt`

Files
- `q_learning.py`: Tabular Q-Learning agent with state discretization.
- `dqn.py`: DQN implementation (PyTorch) with replay buffer and target network.
- `train_qlearning.py`: Train script for Q-Learning, saves Q-table and rewards CSV.
- `train_dqn.py`: Train script for DQN, saves model `.pth` and rewards CSV.
- `utils.py`: Helper functions for seeding, saving and loading.
- `plots.py`: Plotting utilities for learning curves and comparisons.
- `results/`: Output directories for results and figures.

How to run
1. Activate environment: `conda activate RL_v1`
2. From project directory run (examples):

```bash
python -m project.train_qlearning
python -m project.train_dqn
```

Experimental design and report assets
- The `report_assets/` folder contains `figures/` and `tables/` for plots and CSV summaries.

Theory (short)

## Markov Decision Process (MDP)

- State space `S`: continuous 4D vector (cart position, cart velocity, pole angle, pole angular velocity).
- Action space `A`: discrete {0 (left), 1 (right)}.
- Transition probability `P`: environment dynamics (unknown deterministic/stochastic mapping from (s,a) to s').
- Reward function `R`: +1 for each timestep survived.
- Discount factor `γ`: scales future rewards in return computation.

## Temporal Difference Error

The TD error for Q-Learning (one-step) is:
$$\delta = r + \gamma \max_{a'} Q(s',a') - Q(s,a)$$

This error drives the Q-table updates.

## Off-policy learning

Q-Learning is off-policy because it updates estimates using the greedy action \(\max_{a'} Q(s',a')\) regardless of the action taken in the behavior policy.

## Exploration vs Exploitation

We use epsilon-greedy where with probability epsilon a random action is chosen; otherwise the greedy action is used. Epsilon decays over episodes to shift from exploration to exploitation.

## Comparison Between Q-Learning and DQN

- Tabular vs neural approximation: Q-Learning stores explicit table entries; DQN uses a neural net to approximate Q-values.
- Scalability: DQN scales to large/continuous spaces; tabular Q-Learning does not.
- Memory: tabular requires storage proportional to discretized state-action pairs; DQN uses replay buffer plus network weights.
- Convergence & stability: DQN requires tricks (replay buffer, target network, careful hyperparams) to achieve stable learning.
- Continuous states: DQN naturally handles continuous states; tabular requires discretization which loses fidelity.

Notes
- The code is structured for clarity and reproducibility. Use fixed seeds and save CSVs/models for later analysis.
