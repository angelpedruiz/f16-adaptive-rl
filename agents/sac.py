import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as pyd
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from typing import Optional

# Define Transition at module level for pickling
Transition = namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

class Actor(nn.Module):
    """
    Diagonal Gaussian policy with tanh squashing (for SAC).
    Provides both stochastic and deterministic actions.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int], log_std_bounds: tuple[float, float] = (-20, 2)):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        
        # Build MLP
        layers = []
        input_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 2*act_dim)) # last layer outputs mean and log_std
        self.trunk = nn.Sequential(*layers)
        self.apply(self._weights_init_) # weight initialization
        
    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        
        # Constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        
        std = log_std.exp()
        dist = pyd.Normal(mu, std)
        
        # Wrap in TanhTransform to ensure actions are in [-1, 1]
        dist = pyd.TransformedDistribution(dist, pyd.TanhTransform(cache_size=1))
        dist = pyd.Independent(dist, 1)  # treat action dims jointly 
        return dist

    def sample(self, obs):
        '''Sample action and log-prob'''
        dist = self.forward(obs)
        action = dist.rsample()  # reparameterization trick
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob
    
    def deterministic(self, obs):
        '''Deterministic action (for evaluation)'''
        dist = self.forward(obs)
        return dist.mean
    
    @staticmethod
    def _weights_init_(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0)
    
class DoubleQCritic(nn.Module):
    """
    Double Q-network for SAC.
    Accepts state + action as input and returns two Q-values.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]):
        super().__init__()
        
        # Build Q1 network
        layers = []
        input_dim = obs_dim + act_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1)) # last layer outputs Q-value
        self.Q1 = nn.Sequential(*layers)
        
        # Build Q2 network
        layers = []
        input_dim = obs_dim + act_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1)) # last layer outputs Q-value
        self.Q2 = nn.Sequential(*layers)
        
        self.outputs = dict() # for debugging
        self.apply(self._weights_init_) # weight initialization
        
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        self.outputs['q1'] = q1
        self.outputs['q2'] = q2
        return q1, q2
    
    @staticmethod
    def _weights_init_(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0)
        

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: str):
        self.capacity: int = capacity
        self.device: str = device

        # Convert dims to shapes internally
        obs_shape: tuple[int, ...] = (obs_dim,)
        action_shape: tuple[int, ...] = (act_dim,)

        # Pre-allocate storage
        self.obses: np.ndarray = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses: np.ndarray = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions: np.ndarray = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards: np.ndarray = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones: np.ndarray = np.empty((capacity, 1), dtype=np.float32)

        self.idx: int = 0
        self.full: bool = False

    def push(self, obs: np.ndarray, action: np.ndarray, reward: float,
             done: bool, next_obs: np.ndarray) -> None:
        """Add a new transition to the buffer."""
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        self.not_dones[self.idx] = 1.0 - float(done)
        np.copyto(self.next_obses[self.idx], next_obs)

        # Update index
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor]:
        """Sample a batch as PyTorch tensors."""
        max_idx: int = self.capacity if self.full else self.idx
        idxs: np.ndarray = np.random.randint(0, max_idx, size=batch_size)

        observations: torch.Tensor = torch.as_tensor(self.obses[idxs], dtype=torch.float32, device=self.device)
        actions: torch.Tensor = torch.as_tensor(self.actions[idxs], dtype=torch.float32, device=self.device)
        rewards: torch.Tensor = torch.as_tensor(self.rewards[idxs], dtype=torch.float32, device=self.device)
        not_dones: torch.Tensor = torch.as_tensor(self.not_dones[idxs], dtype=torch.float32, device=self.device)
        next_observations: torch.Tensor = torch.as_tensor(self.next_obses[idxs], dtype=torch.float32, device=self.device)

        return observations, actions, rewards, not_dones, next_observations

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx




class SACAgent:
    def __init__(
    self,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: list[int],
    batch_size: int = 64,
    memory_size: int = 10000,
    actor_lr: float = 0.001,
    critic_lr: float = 0.001,
    alpha_lr: float = 0.001,
    init_temp: float = 0.1,
    gamma: float = 0.99,
    tau: float = 0.01,
    learnable_temp: bool = True,
    critic_target_update_freq: int = 1,
    actor_update_freq: int = 1,
    device: str = "cuda",
    env: Optional[object] = None):

        self.device = device
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.init_temp = init_temp
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.tau = tau  # For soft update of target parameters
        self.env = env  # Store environment for action sampling
        self.learnable_temp = learnable_temp
        self.critic_target_update_freq = critic_target_update_freq
        self.actor_update_freq = actor_update_freq
        self.batch_size = batch_size
        
        self.target_entropy = -act_dim  # Target entropy is -|A|
        
        # Temperature (alpha)
        self.log_alpha = torch.tensor(np.log(self.init_temp), dtype=torch.float32,
                                      requires_grad=self.learnable_temp,
                                      device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Actor and Critic Networks
        self.actor = Actor(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        
        self.critic = DoubleQCritic(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.critic_target = DoubleQCritic(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())  # Initialize target network
        
        # Optimizers
        self.critic_optimizer = optim.Adam(list(self.critic.parameters()), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, memory_size, self.device)
        self.learn_step = 0
        
        print('Device:', self.device)
        

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval() # set to eval mode so that behaviour is deterministic
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs_tensor).cpu().data.numpy().flatten()
            else:
                action, _ = self.actor.sample(obs_tensor)
        self.actor.train() # set back to train mode
        action = action.cpu().data.numpy().flatten()
        return np.clip(action, -1, 1)  # Assuming action space is normalized between -1 and 1


    def update(self, obs, action, reward, terminated, next_obs):
        # Store transition [np.ndarray, np.ndarray, float, bool, np.ndarray] in replay buffer
        self.replay_buffer.push(obs, action, reward, terminated, next_obs)

        # Don't update until we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 

        # Sample batch
        obs_batch, action_batch, reward_batch, not_dones_batch, next_obs_batch = self.replay_buffer.sample(self.batch_size)
        
        # ------------------------ Critic Update ------------------------ #
        with torch.no_grad():
            # Next action and log-prob form target actor
            next_action, next_log_prob = self.actor.sample(next_obs_batch)
            target_q1, target_q2 = self.critic_target(next_obs_batch, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_q = reward_batch + self.gamma * target_v * not_dones_batch

        # Compute current Q estimates
        current_q1, current_q2 = self.critic(obs_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ------------------------ Actor Update ------------------------ #
        # Update actor and temperature
        if self.learn_step % self.actor_update_freq == 0:
            action, log_prob = self.actor.sample(obs_batch)
            actor_q1, actor_q2 = self.critic(obs_batch, action)
            actor_q = torch.min(actor_q1, actor_q2)
            actor_loss = (self.alpha.detach() * log_prob - actor_q).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            if self.learnable_temp:
                self.log_alpha_optimizer.zero_grad()
                alpha_loss = (self.alpha *
                            (-log_prob - self.target_entropy).detach()).mean()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
            
        # Update target networks
        if self.learn_step % self.critic_target_update_freq == 0:  
            # Soft update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        self.learn_step += 1
        

    def get_brain(self):
        pass
    
    def load_brain(self, brain_dict):
        pass 

    @property
    def alpha(self):
        return self.log_alpha.exp()