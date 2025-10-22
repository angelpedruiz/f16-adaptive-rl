'''
(HDP) Heuristic Dynamic Programming Agent. Uses a neural network for the model.
'''
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

class Actor(nn.Module):
    ''' Actor Network for HDP: pi: x_t -> u_t. Trained to maximize value V(x_t)'''
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            nn.init.xavier_uniform_(layers[-2].weight)  # Xavier initialization for weights
            nn.init.zeros_(layers[-2].bias)           # Zero initialization for biases
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
    ''' Critic Network for HDP: V: x_t -> V_hat (value/reward-to-go) [float]'''
    def __init__(self, obs_dim: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            nn.init.xavier_uniform_(layers[-2].weight)  # Xavier initialization for weights
            nn.init.zeros_(layers[-2].bias)           # Zero initialization for biases
            prev_size = hidden_size

        # Output layer with better initialization for negative V-values
        output_layer = nn.Linear(prev_size, 1)
        # Initialize output layer with small weights to start near zero
        nn.init.uniform_(output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(output_layer.bias, -3e-3, 3e-3)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
class Model(nn.Module):
    ''' Model Network for HDP: x_hat: (x_t, u_t) -> x_next_pred'''
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev_size = obs_dim + act_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            nn.init.xavier_uniform_(layers[-2].weight)  # Xavier initialization for weights
            nn.init.zeros_(layers[-2].bias)           # Zero initialization for biases
            prev_size = hidden_size

        # Output layer
        output_layer = nn.Linear(prev_size, obs_dim)
        nn.init.uniform_(output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(output_layer.bias, -3e-3, 3e-3)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)
        
    def forward(self, obs_act: torch.Tensor) -> torch.Tensor:
        return self.model(obs_act)
    
class HDPAgent():
    def __init__(self, obs_space: gym.Space, act_space: gym.Space, gamma: float, hidden_sizes: dict[str, list[int]], learning_rates: dict[str, float]):
        
        # Spaces
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]
        self.act_low = act_space.low
        self.act_high = act_space.high
        
        # Hyperparameters
        self.gamma = gamma

        # Initialize Actor, Critic, and Model networks
        self.actor = Actor(self.obs_dim, self.act_dim, hidden_sizes['actor'])
        self.critic = Critic(self.obs_dim, hidden_sizes['critic'])
        self.model = Model(self.obs_dim, self.act_dim, hidden_sizes['model'])

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rates['actor'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rates['critic'])
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rates['model'])
        
        # Memory
        self.prev_obs = None
        self.prev_reward = None

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from actor network given observation and scale to problem dimensions with clipping."""
        obs = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            action = self.actor(obs)
        action = action.squeeze(0).numpy()  # Remove batch dimension
        # Scale action to environment's action space
        scaled_action = self.act_low + (action + 1.0) * 0.5 * (self.act_high - self.act_low)
        return np.clip(scaled_action, self.act_low, self.act_high)
    
    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, next_obs: np.ndarray) -> dict:
        """Update Actor, Critic, and Model networks based on transition tuple."""
        obs = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
        action = torch.FloatTensor(action).unsqueeze(0) # u_t
        reward = torch.FloatTensor([reward]).unsqueeze(0) 
        next_obs = torch.FloatTensor(next_obs).unsqueeze(0)
        

        # ------- Update Model -------
        model_input = torch.cat([obs, action], dim=-1) # (x_t, u_t)
        next_obs_pred = self.model(model_input) # x_{t+1}_pred
        model_loss = 0.5 * (next_obs - next_obs_pred).pow(2).mean() # MSE loss

        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        # ------- Update Critic -------
        with torch.no_grad():
            target_value = self.prev_reward + self.gamma * self.critic(obs) # r_{t-1} + γ*V(x_t)
        value_pred = self.critic(self.prev_obs) # V(x_{t-1})
        td_error = target_value - value_pred
        critic_loss = 0.5 * (td_error).pow(2).mean() # MSE loss

        self.critic_optimizer.zero_grad() 
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------- Update Actor -------
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.critic.parameters():
            p.requires_grad = False

        actor_action = self.actor(obs)  # pi(x_t)
        next_obs_pred = self.model(torch.cat([obs, actor_action], dim=-1))  # xpred_{t+1}(x_t, pi(x_t))
        actor_loss = -self.critic(next_obs_pred).mean()  # -V(xpred_{t+1}(x_t, pi(x_t)))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.critic.parameters():
            p.requires_grad = True
            
        # ------- Update memory -------
        self.prev_obs = obs.detach()
        self.prev_reward = reward.detach()

        with torch.no_grad():
            return {
                'critic_error': td_error.abs().item(),
                'model_error': (next_obs - next_obs_pred).abs().mean().item(),
                'model_prediction': next_obs_pred.detach().squeeze(0).numpy(),
                'true_state': next_obs.detach().squeeze(0).numpy(),
                'losses': {
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'model_loss': model_loss.item()
                },
                'weights_norm': {
                    'actor': np.linalg.norm(np.concatenate([p.data.cpu().numpy().flatten() for p in self.actor.parameters()])),
                    'critic': np.linalg.norm(np.concatenate([p.data.cpu().numpy().flatten() for p in self.critic.parameters()])),
                    'model': np.linalg.norm(np.concatenate([p.data.cpu().numpy().flatten() for p in self.model.parameters()]))
                },
                'weights_update_norm': {
                    'actor': np.linalg.norm(np.concatenate([p.grad.data.cpu().numpy().flatten() for p in self.actor.parameters() if p.grad is not None])),
                    'critic': np.linalg.norm(np.concatenate([p.grad.data.cpu().numpy().flatten() for p in self.critic.parameters() if p.grad is not None])),
                    'model': np.linalg.norm(np.concatenate([p.grad.data.cpu().numpy().flatten() for p in self.model.parameters() if p.grad is not None]))
                },
                'gradients_norm': {
                    'actor': np.linalg.norm(np.concatenate([p.grad.data.cpu().numpy().flatten() for p in self.actor.parameters() if p.grad is not None])),
                    'critic': np.linalg.norm(np.concatenate([p.grad.data.cpu().numpy().flatten() for p in self.critic.parameters() if p.grad is not None])),
                    'model': np.linalg.norm(np.concatenate([p.grad.data.cpu().numpy().flatten() for p in self.model.parameters() if p.grad is not None]))
                }  
            }
