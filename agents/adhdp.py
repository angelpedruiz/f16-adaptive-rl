import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=32, num_layers=2, act_limit=1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_dim, hidden_dim))  # input layer
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        x = obs
        for layer in self.layers:
            x = F.relu(layer(x))
        action = torch.tanh(self.out(x)) * self.act_limit
        return action
    


class CriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_dim + act_dim, hidden_dim))  # input layer
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x))
        q_value = self.out(x)
        return torch.squeeze(q_value, -1)  # Critical to ensure q has right shape.


class ADHDPAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        action_low,
        action_high,
        hidden_dim=32,
        num_layers=2,
        act_limit=1.0,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        noise=True,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma

        # Save bounds for rescaling
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

        # Networks
        self.actor = ActorNet(obs_dim, act_dim, hidden_dim, num_layers, act_limit).to(
            device
        )
        self.critic = CriticNet(obs_dim, act_dim, hidden_dim, num_layers).to(device)
        self.noise = noise

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Logging / tracking
        self.training_error = []

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw_action = self.actor(obs).cpu().numpy().flatten()

        # Rescale raw action to physical bounds
        physical_action = ((raw_action + 1) / 2) * (
            self.action_high.cpu().numpy() - self.action_low.cpu().numpy()
        ) + self.action_low.cpu().numpy()

        # Add Gaussian noise if enabled
        if getattr(self, "noise", False):
            noise = np.random.normal(0, 0.5, size=physical_action.shape)
            physical_action = np.clip(
                physical_action + noise,
                self.action_low.cpu().numpy(),
                self.action_high.cpu().numpy(),
            )

        return physical_action

    def update(self, obs, action, cost, terminated, next_obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        cost = torch.tensor(cost, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        # ------------------------
        # Critic Update
        # ------------------------
        next_action = self.actor(next_obs)
        td_target = cost + self.gamma * self.critic(next_obs, next_action)
        td_loss = nn.MSELoss()(self.critic(obs, action), td_target.detach())

        self.critic_opt.zero_grad()
        td_loss.backward()
        self.critic_opt.step()

        # Log TD error
        self.training_error.append(td_loss.item())

        # ------------------------
        # Actor Update
        # ------------------------
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

    def reset_training_log(self):
        self.training_error = []

    def get_training_log(self):
        return self.training_error

    def get_brain(self):
        """
        Returns agent's brain as a dictionary for serialization.
        Includes network weights, optimizer states, hyperparameters, and training log.
        """
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
        """
        Load agent's brain from a dictionary.
        """
        self.actor.load_state_dict(brain_dict["actor_state_dict"])
        self.critic.load_state_dict(brain_dict["critic_state_dict"])
        self.actor_opt.load_state_dict(brain_dict["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(brain_dict["critic_opt_state_dict"])
        self.gamma = brain_dict["gamma"]
        self.training_error = brain_dict.get("training_error", [])
        self.action_low = torch.tensor(
            brain_dict["action_low"], dtype=torch.float32, device=self.device
        )
        self.action_high = torch.tensor(
            brain_dict["action_high"], dtype=torch.float32, device=self.device
        )
