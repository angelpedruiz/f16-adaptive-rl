import numpy as np
import torch
import torch.nn as nn
"""
Adaptive Heuristic Dynamic Programming (ADHDP) implementation in PyTorch.

This module defines an actor–critic architecture where:
- The **Critic** network approximates the value function V(S, A) (reward-to-go), trained using
  a Temporal Difference (TD) error derived from the Bellman equation.
- The **Actor** network maps states S to actions A, trained to maximize the predicted
  long-term reward as estimated by the critic.

Learning objective:
    Critic loss: L_c = 0.5 * [Q(S_t, A_t) - (R_t + γ(1 - done) * Q(S_{t+1}, A_{t+1}))]^2
    Actor  loss: L_a = -Q(S_t, A_t)  (negative to maximize via gradient descent)

This corresponds to the action-dependent HDP algorithm, where both actor and critic
are updated using backpropagation through differentiable neural networks.
"""


class Actor(nn.Module):
    ''' Actor Network for ADHDP: St -> At. Trained to maximize value Q(s,a)'''
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer with small initialization for stable initial policy
        output_layer = nn.Linear(prev_size, act_dim)
        nn.init.uniform_(output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(output_layer.bias, -3e-3, 3e-3)
        layers.append(output_layer)
        layers.append(nn.Tanh()) # Output in [-1, 1] for continuous action space

        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

class Critic(nn.Module):
    ''' Critic Network for ADHDP: (St, At) -> Q(s,a) (value/reward-to-go) [float]'''
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev_size = obs_dim + act_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer with better initialization for negative Q-values
        output_layer = nn.Linear(prev_size, 1)
        # Initialize output layer with small weights to start near zero
        nn.init.uniform_(output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(output_layer.bias, -3e-3, 3e-3)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([obs, act], dim=-1))


class ADHDPAgent():
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int], actor_lr: float, critic_lr: float, gamma: float, device: str, action_low: np.ndarray = None, action_high: np.ndarray = None):
        self.device = device
        self.gamma = gamma

        self.actor = Actor(obs_dim, act_dim, hidden_sizes).to(device)
        self.critic = Critic(obs_dim, act_dim, hidden_sizes).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Store action space bounds for scaling
        self.action_low = action_low if action_low is not None else -np.ones(act_dim)
        self.action_high = action_high if action_high is not None else np.ones(act_dim)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action from actor network and scale from [-1, 1] to action space bounds.

        Args:
            obs: Observation from environment

        Returns:
            action: Scaled action in the environment's action space
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            action_tensor = self.actor(obs_tensor)  # Output in [-1, 1] due to tanh
        action_normalized = action_tensor.cpu().numpy().flatten()  # Remove batch dimension and convert to numpy

        # Scale from [-1, 1] to [action_low, action_high]
        action_scaled = self.action_low + (action_normalized + 1.0) * 0.5 * (self.action_high - self.action_low)

        return action_scaled
    
    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, next_obs: np.ndarray) -> None:
        """
        Update actor and critic networks using the ADHDP algorithm.

        The critic learns the value function Q(s,a) using the Bellman equation:
            Q(s,a) = reward + γ * (1 - done) * Q(s',a')

        The actor is trained to maximize the expected value:
            max_a Q(s,a) => min_a -Q(s,a)

        Args:
            obs: Current observation
            action: Action taken (already scaled to environment bounds)
            reward: Reward received (higher is better)
            terminated: Whether episode terminated
            next_obs: Next observation
        """
        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        done = torch.tensor(terminated, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # ===== Critic Update =====
        # Compute TD target: reward + γ * Q(s', a')
        with torch.no_grad():
            next_action = self.actor(next_obs) # A_t+1
            Q_next = self.critic(next_obs, next_action) # Q(s', a')
            td_target = reward + self.gamma * (1 - done) * Q_next

        # Compute TD error
        Q_pred = self.critic(obs, action) # forward pass to get Q(s, a)
        td_error = td_target - Q_pred
        critic_loss = 0.5 * ((td_error) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ===== Actor Update =====
        # Maximize Q(s, a) by minimizing -Q(s, a)
        actor_loss = -self.critic(obs, self.actor(obs)).mean() # -Q(s, a)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        

