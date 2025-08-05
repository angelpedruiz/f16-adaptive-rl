from collections import defaultdict, deque
import gymnasium as gym
import numpy as np
from torch import nn
import torch.nn.functional as F


class Agent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        obs_discretizer=None,
        action_discretizer=None,
        brain_state: dict = None,
    ):
        self.env = env
        self.agent_obs_space = obs_discretizer
        if action_discretizer is None:
            self.agent_action_space = env.action_space
            self.num_actions = env.action_space.n
        else:
            self.agent_action_space = action_discretizer
            self.num_actions = np.prod(self.agent_action_space.space.nvec)

        if brain_state:
            self.q_values = brain_state.get("q_values", defaultdict(lambda: np.zeros(self.num_actions)))
            self.lr = brain_state.get("learning_rate", learning_rate)
            self.discount_factor = brain_state.get("discount_factor", discount_factor)
            self.epsilon = brain_state.get("epsilon", initial_epsilon)
            self.training_error = brain_state.get("training_error", [])
        else:
            self.q_values = defaultdict(lambda: np.zeros(self.num_actions))
            self.lr = learning_rate
            self.discount_factor = discount_factor
            self.epsilon = initial_epsilon
            self.training_error = []

        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def get_action(self, obs: tuple) -> tuple:
        discretized_obs = self.agent_obs_space.discretize(obs)

        if np.random.rand() < self.epsilon:
            action = tuple(self.agent_action_space.space.sample())
        else:
            q_vals = self.q_values[discretized_obs]
            best_actions = np.flatnonzero(q_vals == q_vals.max())
            action = (np.random.choice(best_actions),)

        undiscretized_action = self.agent_action_space.undiscretize(action)
        return undiscretized_action

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
