import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from typing import Optional
import gymnasium as gym
from collections import namedtuple, deque
import random

# Define Transition at module level for pickling
Transition = namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

class Actor(nn.Module):
    ''' Actor Network for TD3 state -> action'''
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.out = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return F.tanh(self.out(x))  # Assuming action space is continuous
    
class Critic(nn.Module):
    ''' Critic Network for TD3 state, state + action -> Q-value'''
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim + action_dim, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.out = nn.Linear(hidden_sizes[-1], 1) 

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)

class ReplayMemory:
    def __init__(self, capacity: int, device: str):
        self.memory: deque = deque([], maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, done, next_state):
        self.memory.append(Transition(state, action, reward, done, next_state))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.memory, batch_size) if batch_size < len(self.memory) else list(self.memory)
        observations, actions, rewards, dones, next_observations = zip(*batch)

        observations = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, device=self.device)


        return observations, actions, rewards, dones, next_observations

    def __len__(self) -> int:
        return len(self.memory)


class TD3Agent:
    def __init__(
    self,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: list[int],
    batch_size: int = 64,
    memory_size: int = 10000,
    actor_lr: float = 0.001,
    critic_lr: float = 0.001,
    gamma: float = 0.99,
    tau: float = 0.01,
    exploration_noise_start: float = 0.1,
    exploration_noise_decay: float = 0.95,
    exploration_noise_min: float = 0.01,
    target_noise_std: float = 0.2,
    target_noise_clip: float = 0.5,
    max_grad_norm: float = 1.0,
    policy_delay: int = 2,
    device: str = "cuda",
    env: Optional[object] = None):

        self.device = device
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.exploration_noise = exploration_noise_start
        self.exploration_noise_decay = exploration_noise_decay
        self.exploration_noise_min = exploration_noise_min
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.max_grad_norm = max_grad_norm # For gradient clipping
        self.policy_delay = policy_delay  # Frequency of policy updates
        self.tau = tau  # For soft update of target parameters
        self.env = env  # Store environment for action sampling
        
        self.learn_step = 0  # To track when to update policy
        
        # Replay Buffer
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replay_buffer = ReplayMemory(self.memory_size, self.device)

        # Actor and Critic Networks
        self.actor = Actor(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict()) # target policy should be a clone of the policy at the start
        self.critic_1 = Critic(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic_2 = Critic(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic_target_1 = Critic(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic_target_2 = Critic(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        
        # Optimizers and loss function
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=self.critic_lr) # manages both critics
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        
        print('Device:', self.device)
        

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval() # set to eval mode so that behaviour is deterministic
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().data.numpy().flatten()
        self.actor.train() # set back to train mode
        action += np.random.normal(0, self.exploration_noise, size=self.act_dim)  # Add exploration noise
        return np.clip(action, -1, 1)  # Assuming action space is normalized between -1 and 1


    def update(self, obs, action, reward, terminated, next_obs):
        # Store transition [np.ndarray, np.ndarray, float, bool, np.ndarray] in replay buffer
        self.replay_buffer.push(obs, action, reward, terminated, next_obs)

        # Don't update until we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to update

        # Sample batch
        obs_batch, action_batch, reward_batch, dones_batch, next_obs_batch = self.replay_buffer.sample(self.batch_size)

        # Compute target critic Q values
        with torch.no_grad():
            noise = (torch.randn_like(action_batch) * self.target_noise_std).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = (self.actor_target(next_obs_batch) + noise).clamp(-1, 1)
            target_q1 = self.critic_target_1(next_obs_batch, next_action)
            target_q2 = self.critic_target_2(next_obs_batch, next_action)
            target_q = reward_batch + self.gamma * torch.min(target_q1, target_q2) * (1 - dones_batch)
            
        # Compute critic Q values and loss
        q1 = self.critic_1(obs_batch, action_batch)
        q2 = self.critic_2(obs_batch, action_batch)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Update policy and soft update target networks
        if self.learn_step % self.policy_delay == 0:
            actor_loss = -self.critic_1(obs_batch, self.actor(obs_batch)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        self.learn_step += 1
        
        

    def decay_exploration(self):
        """Decay exploration noise after each episode"""
        self.exploration_noise = max(self.exploration_noise_min, self.exploration_noise * self.exploration_noise_decay)

    def get_brain(self):
        pass
    
    def load_brain(self, brain_dict):
        pass 
