import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------- Actor Network ----------------
class ActorNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=32, num_layers=2, act_limit=1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_dim, hidden_dim))  # input layer
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, act_dim)  # output layer
        self.act_limit = act_limit

    def forward(self, obs):
        x = obs
        for layer in self.layers:
            x = F.relu(layer(x))
        action = torch.tanh(self.out(x)) * self.act_limit  # scale output
        return action


# ---------------- Critic Network ----------------
class CriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_dim + act_dim, hidden_dim))  # input layer
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, 1)  # Q-value output

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)  # combine state and action
        for layer in self.layers:
            x = F.relu(layer(x))
        q_value = self.out(x)
        return torch.squeeze(q_value, -1)  # ensure proper shape


# ---------------- ADHDP Agent ----------------
class ADHDPAgent:
    def __init__(self, obs_dim, act_dim, action_low, action_high,
                 hidden_dim=32, num_layers=2, act_limit=1.0,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99,
                 noise=True, device="cpu"):

        self.device = device
        self.gamma = gamma
        self.noise = noise

        # Action bounds
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

        # Networks
        self.actor = ActorNet(obs_dim, act_dim, hidden_dim, num_layers, act_limit).to(device)
        self.critic = CriticNet(obs_dim, 2, hidden_dim, num_layers).to(device)

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Logging
        self.training_error = []
        self.last_action = None

    # Convert scalar actor output to full 2D action
    def _build_full_action(self, raw_scalar, add_noise=True):
        scaled = ((raw_scalar + 1) / 2) * (self.action_high[1] - self.action_low[1]) + self.action_low[1]
        if add_noise and self.noise:
            noise = np.random.normal(0, 0.5)
            scaled = torch.clamp(scaled + noise, self.action_low[1], self.action_high[1])
        return np.array([0.0, scaled.item()])

    # Get action for environment
    def get_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw_scalar = self.actor(obs_tensor)
        physical_action = self._build_full_action(raw_scalar, add_noise=True)
        self.last_action = physical_action
        return physical_action

    # Update actor and critic
    def update(self, obs, action, reward, terminated, next_obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            next_raw = self.actor(next_obs)
            next_action = torch.tensor(self._build_full_action(next_raw, add_noise=False),
                                       dtype=torch.float32, device=self.device).unsqueeze(0)
            next_q = self.critic(next_obs, next_action)
            td_target = reward + (0.0 if terminated else self.gamma * next_q)

        # Critic update
        td_loss = F.mse_loss(self.critic(obs, action), td_target)
        self.critic_opt.zero_grad()
        td_loss.backward()
        self.critic_opt.step()
        self.training_error.append(td_loss.item())

        # Actor update
        raw_actor = self.actor(obs)
        actor_action = torch.tensor(self._build_full_action(raw_actor, add_noise=False),
                                    dtype=torch.float32, device=self.device).unsqueeze(0)
        actor_loss = -self.critic(obs, actor_action).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

    # Logging helpers
    def reset_training_log(self):
        self.training_error = []

    def get_training_log(self):
        return self.training_error

    # Serialization
    def get_brain(self):
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_opt_state_dict": self.actor_opt.state_dict(),
            "critic_opt_state_dict": self.critic_opt.state_dict(),
            "gamma": self.gamma,
            "training_error": self.training_error,
            "action_low": self.action_low.cpu().numpy(),
            "action_high": self.action_high.cpu().numpy(),
        }

    def load_brain(self, brain_dict):
        self.actor.load_state_dict(brain_dict["actor_state_dict"])
        self.critic.load_state_dict(brain_dict["critic_state_dict"])
        self.actor_opt.load_state_dict(brain_dict["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(brain_dict["critic_opt_state_dict"])
        self.gamma = brain_dict["gamma"]
        self.training_error = brain_dict.get("training_error", [])
        self.action_low = torch.tensor(brain_dict["action_low"], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(brain_dict["action_high"], dtype=torch.float32, device=self.device)
