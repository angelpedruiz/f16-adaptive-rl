import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from typing import Optional
import gymnasium as gym
from collections import namedtuple, deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[24, 24]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.out = nn.Linear(hidden_sizes[-1], action_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)

    

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return (
            random.sample(self.memory, batch_size)
            if batch_size < len(self.memory)
            else self.memory
        )

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, obs_dim, act_dim, hidden_sizes: list, batch_size=64, memory_size=10000, learning_rate=0.001, gamma=0.99, tau=0.01, epsilon_start=0.1, epsilon_decay=0.95, epsilon_min=0.01, device="cpu", env=None):
        self.device = device
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau  # For soft update of target parameters
        self.env = env  # Store environment for action sampling
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        assert self.is_discrete, "DQNAgent only supports discrete action spaces"
        
        # Replay Buffer
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replay_buffer = ReplayMemory(self.memory_size)

        # Q-Network
        self.policy_net = DQN(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.target_net = DQN(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        

    def get_action(self, obs: np.ndarray) -> int:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return self.policy_net(obs_tensor).max(1).indices.item()


    def update(self, obs, action, reward, terminated, next_obs):
        
        # Convert to tensors or appropriate types
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long if isinstance(self.env.action_space, gym.spaces.Discrete) else torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = terminated
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Store transition
        self.replay_buffer.push(obs, action, reward, done, next_obs)
        
        # Soft update target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        # Only update if enough samples are available
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = self.replay_buffer.Transition(*zip(*transitions))
        observations = batch.state
        actions = batch.action  
        rewards = batch.reward
        dones = batch.done
        next_observations = batch.next_state

        # Concatenate
        observations_batch = torch.cat(observations)
        actions_batch = torch.cat(actions).unsqueeze(1)
        next_obs_batch = torch.cat(next_observations)
        rewards_batch = torch.cat(rewards)
        dones_batch = torch.tensor(dones, device=self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(observations_batch).gather(1, actions_batch)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_obs_batch).max(dim=1)[0].unsqueeze(1)
        target_q_values = rewards_batch.unsqueeze(1) + self.gamma * next_q_values * (~dones_batch).unsqueeze(1)

        # Compute loss and optimize
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_brain(self):
        pass
    
    def load_brain(self, brain_dict):
        pass 
