import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from typing import Optional
import gymnasium as gym
from collections import namedtuple, deque
import random

class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_dim, hidden_sizes[0], dtype=torch.float32))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], dtype=torch.float32))
        self.out = nn.Linear(hidden_sizes[-1], action_dim, dtype=torch.float32)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return torch.tanh(self.out(x))  # Assuming actions are in range [-1, 1]

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: list):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_dim, hidden_sizes[0], dtype=torch.float32))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], dtype=torch.float32))
        self.out = nn.Linear(hidden_sizes[-1], obs_dim, dtype=torch.float32)  # value gradient
    
    def forward(self, state):
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)  # λ_hat(x)
    
class IncrementalModelRLS:
    def __init__(self, obs_dim: int, act_dim: int, lam: float = 0.99, delta: float = 1.0):
        """
        obs_dim: dimension of state vector
        act_dim: dimension of action vector
        lam: forgetting factor (0.95–1.0, closer to 1 = slower forgetting)
        delta: initial covariance scaling
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.param_dim = obs_dim + act_dim  # size of regressor phi
        
        # Parameter matrix Θ: shape (obs_dim, obs_dim+act_dim)
        self.Theta = np.zeros((obs_dim, self.param_dim), dtype=np.float32)
        
        # Covariance matrix P (big, but manageable for small dims)
        self.P = np.eye(self.param_dim, dtype=np.float32) * delta
        
        self.lam = lam

    def update(self, x: np.ndarray, u: np.ndarray, next_x: np.ndarray):
        dx = (next_x - x).astype(np.float32)
        phi = np.concatenate([x, u]).astype(np.float32)  # regressor vector
        phi = phi.reshape(-1, 1)      # column
        
        # Compute gain
        P_phi = self.P @ phi
        gain = P_phi / (self.lam + phi.T @ P_phi)
        
        # Prediction error
        err = dx.reshape(-1, 1) - self.Theta @ phi
        
        # Update parameters
        self.Theta += err @ gain.T
        
        # Update covariance
        self.P = (self.P - gain @ phi.T @ self.P) / self.lam

    def get_jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        # Split Theta into A and B parts
        A = self.Theta[:, :self.obs_dim].astype(np.float32)
        B = self.Theta[:, self.obs_dim:].astype(np.float32)
        return A, B



class IDHPAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.0002,
        gamma: float = 0.99,
        device: str = "cpu",
        env=None,
        rls_lam: float = 0.99,
        rls_delta: float = 1.0,
        tau: float = 0.001,
        reward_scale: float = 1.0,
    ):
        self.device = device
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.env = env
        self.tau = tau
        self.reward_scale = reward_scale
        
        # Networks
        self.actor = Actor(obs_dim, act_dim, hidden_sizes).to(device)
        self.critic = Critic(obs_dim, hidden_sizes).to(device)
        self.critic_target = Critic(obs_dim, hidden_sizes).to(device)
        self.model = IncrementalModelRLS(obs_dim, act_dim, lam=rls_lam, delta=rls_delta)
        
        # Initialize target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Previous states for model updates
        self.prev_state = None
        self.prev_action = None

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        return action.astype(np.float32)
    
    def update(self, obs, action, reward, terminated, next_obs, info=None):
        # -------------------------
        # 1. Update incremental model (RLS) with state differences
        # -------------------------
        if self.prev_state is not None and self.prev_action is not None:
            # Use state differences like in reference
            state_diff = obs - self.prev_state
            action_diff = action - self.prev_action  
            next_state_diff = next_obs - obs
            self.model.update(state_diff, action_diff, next_state_diff)
        
        # Store current state/action for next update
        self.prev_state = obs.copy()
        self.prev_action = action.copy()
        
        A_np, B_np = self.model.get_jacobians()
        
        # Convert A, B to tensors
        F = torch.tensor(A_np, dtype=torch.float32, device=self.device)  # F matrix
        G = torch.tensor(B_np, dtype=torch.float32, device=self.device)  # G matrix
        
        # -------------------------
        # 2. Convert states/actions to tensors
        # -------------------------
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # -------------------------
        # 3. Compute reward gradient (quadratic cost)
        # -------------------------
        # Simple quadratic cost gradient: 2*Q*x (assuming Q=I)
        reward_grad = 2 * obs_t * self.reward_scale  # shape (1, obs_dim)
        
        # -------------------------
        # 4. Forward pass through networks
        # -------------------------
        with torch.no_grad():
            lambda_next = self.critic_target(next_obs_t)  # Use target network
        
        # Current critic output
        lambda_current = self.critic(obs_t)
        
        # Actor output
        action_pred = self.actor(obs_t)
        
        # -------------------------
        # 5. Compute policy gradient w.r.t state (dpidx)
        # -------------------------
        obs_t.requires_grad_(True)
        action_for_grad = self.actor(obs_t)
        
        # Compute Jacobian matrix: dpidx[i,j] = ∂π_i/∂x_j
        dpidx_list = []
        for i in range(self.act_dim):
            grad_outputs = torch.zeros_like(action_for_grad)
            grad_outputs[0, i] = 1.0
            grad = torch.autograd.grad(action_for_grad, obs_t, grad_outputs=grad_outputs, 
                                     retain_graph=True, create_graph=True)[0]
            dpidx_list.append(grad.squeeze())
        dpidx = torch.stack(dpidx_list, dim=0)  # shape (act_dim, obs_dim)
        
        # -------------------------
        # 6. Critic update (proper IDHP equation)
        # -------------------------
        self.critic_optimizer.zero_grad()
        
        # IDHP critic equation: td_err_ds = (reward_grad + γ*λ_next) @ (F + G @ dpidx) - λ
        with torch.no_grad():
            td_target_term = (reward_grad + self.gamma * lambda_next) @ (F + G @ dpidx)
        td_error = td_target_term - lambda_current
        
        # Use negative td_error as gradient (following reference line 326)
        critic_grad = -td_error
        # Apply gradients manually using output_gradients approach - ensure shape matches
        lambda_current.backward(gradient=critic_grad)
        self.critic_optimizer.step()
        
        # -------------------------
        # 7. Actor update (proper IDHP gradient) 
        # -------------------------
        self.actor_optimizer.zero_grad()
        
        # Actor gradient: -(reward_grad + γ*λ_next) @ G (reference line 315)
        with torch.no_grad():
            actor_grad_coeff = -(reward_grad + self.gamma * lambda_next) @ G
        
        # Apply gradients manually using output_gradients approach - ensure shape matches
        action_pred = self.actor(obs_t)
        action_pred.backward(gradient=actor_grad_coeff)
        self.actor_optimizer.step()
        
        # -------------------------
        # 8. Soft update target network
        # -------------------------
        self._soft_update_target()
        
        obs_t.requires_grad_(False)
    
    def _soft_update_target(self):
        """Soft update of target network parameters"""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        