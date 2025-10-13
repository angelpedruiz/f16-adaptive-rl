import numpy as np
import torch
import torch.nn as nn
"""
Adaptive Heuristic Dynamic Programming (ADHDP) implementation in PyTorch.

This module defines an actor–critic architecture where:
- The **Critic** network approximates the cost-to-go function J(S, A), trained using
  a Temporal Difference (TD) error derived from the Bellman equation.
- The **Actor** network maps states S to actions A, trained to minimize the predicted
  long-term cost (or equivalently maximize the reward-to-go) as estimated by the critic.

Learning objective:
    Critic loss: L_c = 0.5 * [J(S_t, A_t) - (R_t + γ(1 - done) * J(S_{t+1}, A_{t+1}))]^2
    Actor  loss: L_a = -J(S_t, A_t)

This corresponds to the action-dependent HDP algorithm, where both actor and critic
are updated using backpropagation through differentiable neural networks.
"""


class Actor(nn.Module):
    ''' Actor Network for HDP St -> At. Loss: J (cost-to-go)'''
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, act_dim)) # Output layer
        layers.append(nn.Tanh()) # Assuming action space is continuous   
        self.model = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
        
class Critic(nn.Module):
    ''' Critic Network for HDP St, At -> Jt (cost-to-go) [float]'''
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev_size = obs_dim + act_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1)) # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([obs, act], dim=-1))


class ADHDPAgent():
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int], actor_lr: float, critic_lr: float, gamma: float, device: str):
        self.device = device
        self.gamma = gamma

        self.actor = Actor(obs_dim, act_dim, hidden_sizes).to(device)
        self.critic = Critic(obs_dim, act_dim, hidden_sizes).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    def get_action(self, obs: np.ndarray) -> np.ndarray: # At [-1, 1]
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            action_tensor = self.actor(obs_tensor)
        return action_tensor.cpu().numpy().flatten()  # Remove batch dimension and convert to numpy
    
    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, next_obs: np.ndarray) -> None:
        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        done = torch.tensor(terminated, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # ===== Critic Update =====
        next_action = self.actor(next_obs)
        J_pred = self.critic(obs, action) # forward pass
        J_next = self.critic(next_obs, next_action)
        critic_error = J_pred - (reward + self.gamma * (1 - done) * J_next)
        critic_loss = 0.5 * (critic_error ** 2).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ===== Actor Update =====
        actor_loss = -self.critic(obs, self.actor(obs)).mean() # forward pass maximizing Gt (discounted return) = -Jt
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        

