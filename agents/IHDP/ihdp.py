
'''
(IHDP) Incremental Heuristic Dynamic Programming Agent. Uses (RLS) for the model.
'''
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym



class Actor(nn.Module):
    ''' Actor Network for HDP with LayerNorm and stable output '''
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev_size = obs_dim

        # Hidden layers
        for hidden_size in hidden_sizes:
            linear = nn.Linear(prev_size, hidden_size)
            nn.init.xavier_uniform_(linear.weight)  # Xavier init for hidden layers
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.LayerNorm(hidden_size))  # normalize hidden activations
            layers.append(nn.Tanh())
            prev_size = hidden_size

        # Output layer with small uniform initialization
        output_layer = nn.Linear(prev_size, act_dim)
        nn.init.uniform_(output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(output_layer.bias, -3e-3, 3e-3)
        layers.append(output_layer)
        layers.append(nn.Tanh())  # output in [-1,1]

        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.model(obs)
        return x


    
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


class RLSModel():
    """
    Least Squares Model using Recursive Least Squares (RLS) with forgetting factor.

    Estimates a linear approximation of system dynamics:
        dx_{t+1} = F @ dx_t + G @ du_t

    Where:
    - F: state transition Jacobian (obs_dim × obs_dim)
    - G: control sensitivity matrix (obs_dim × act_dim) - the key output

    The G matrix approximates ∂x_{t+1}/∂u_t (derivative of next state w.r.t. control).
    """

    def __init__(self, obs_dim: int, act_dim: int, forgetting_factor: float = 0.99, delta: float = 1.0):
        """
        Initialize the LSModel.

        Args:
            obs_dim: Dimension of observation/state space
            act_dim: Dimension of action space
            forgetting_factor: Forgetting factor (typically 0.95-1.0). Higher = more history retained
            delta: Initial covariance scaling factor (higher = more exploration initially)
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.forgetting_factor = forgetting_factor

        # Parameter matrix Θ: shape (obs_dim, obs_dim + act_dim)
        # Structure: [F | G] where:
        #   F = Θ[:, :obs_dim]     - state transition Jacobian
        #   G = Θ[:, obs_dim:]     - control sensitivity matrix
        self.Theta = np.zeros((obs_dim, obs_dim + act_dim), dtype=np.float32)

        # Covariance matrix P: shape (obs_dim + act_dim, obs_dim + act_dim)
        # Used for RLS numerical stability and adaptation
        self.P = np.eye(obs_dim + act_dim, dtype=np.float32) * delta

    def update(self, x: np.ndarray, u: np.ndarray, next_x: np.ndarray):
        """
        Update the RLS model with new data point.

        Args:
            x: Current state (obs_dim,)
            u: Control input (act_dim,)
            next_x: Next state (obs_dim,)
        """
        # Ensure numpy arrays
        x = np.asarray(x, dtype=np.float32).flatten()
        u = np.asarray(u, dtype=np.float32).flatten()
        next_x = np.asarray(next_x, dtype=np.float32).flatten()

        # State change
        dx = next_x - x  # shape (obs_dim,)

        # Regressor vector: concatenate state and action
        phi = np.concatenate([x, u])  # shape (obs_dim + act_dim,)

        # RLS update equations with forgetting factor
        P_phi = self.P @ phi  # shape (obs_dim + act_dim,)
        denominator = self.forgetting_factor + np.dot(phi, P_phi)  # scalar
        gain = P_phi / denominator  # shape (obs_dim + act_dim,)

        # Prediction error for each state dimension
        pred = self.Theta @ phi  # shape (obs_dim,)
        err = dx - pred  # shape (obs_dim,)

        # Update parameter matrix: Θ = Θ + err @ gain^T
        self.Theta += np.outer(err, gain)

        # Update covariance matrix with forgetting factor
        self.P = (self.P - np.outer(gain, phi) @ self.P) / self.forgetting_factor

    def get_jacobians(self) -> tuple:
        """
        Extract the Jacobian matrices from the parameter matrix.

        Returns:
            F: State transition Jacobian (obs_dim × obs_dim)
            G: Control sensitivity matrix (obs_dim × act_dim)
        """
        F = self.Theta[:, :self.obs_dim]  # First obs_dim columns
        G = self.Theta[:, self.obs_dim:]  # Last act_dim columns
        return F, G

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Predict next state using the learned linear model.

        Args:
            x: Current state (obs_dim,)
            u: Control input (act_dim,)

        Returns:
            Predicted next state (obs_dim,)
        """
        x = np.asarray(x, dtype=np.float32).flatten()
        u = np.asarray(u, dtype=np.float32).flatten()

        F, G = self.get_jacobians()
        dx = F @ x + G @ u
        next_x = x + dx
        return next_x

class IHDPAgent():
    def __init__(self, obs_space: gym.Space, act_space: gym.Space, gamma: float, forgetting_factor: float, initial_covariance: float, hidden_sizes: dict[str, list[int]], learning_rates: dict[str, float]):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.model = RLSModel(self.obs_dim, self.act_dim, forgetting_factor=0.99, delta=1.0)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rates['actor'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rates['critic'])
        
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

    @staticmethod
    def _compute_norm(grad_list):
        """Compute L2 norm of gradients, handling empty lists."""
        if not grad_list:
            return 0.0
        return np.linalg.norm(np.concatenate(grad_list))

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, next_obs: np.ndarray) -> dict:
        """Update Actor, Critic, and Model networks based on a single transition tuple."""

        # --- Convert inputs to torch tensors ---
        obs_torch = torch.FloatTensor(obs).unsqueeze(0)        # shape: (1, obs_dim)
        action_torch = torch.FloatTensor(action).unsqueeze(0) # shape: (1, act_dim)
        reward_torch = torch.FloatTensor([reward]).unsqueeze(0)
        next_obs_torch = torch.FloatTensor(next_obs).unsqueeze(0)

        # ------- Update Model -------
        self.model.update(obs_torch.detach().numpy().squeeze(),
                        action_torch.detach().numpy().squeeze(),
                        next_obs_torch.detach().numpy().squeeze())

        # ------- Update Critic (TD(0) on current transition) -------
        with torch.no_grad():
            target_value = reward_torch + self.gamma * self.critic(next_obs_torch)
        value_pred = self.critic(obs_torch)
        td_error = target_value - value_pred
        critic_loss = 0.5 * td_error.pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------- Update Actor (manual gradient via RLS Jacobian) -------
        # 1️⃣ Forward pass through actor
        actor_action = self.actor(obs_torch)
        actor_action.retain_grad()

        # 2️⃣ Predict next state with RLS model (non-differentiable)
        x_np = obs_torch.detach().squeeze().numpy()
        u_np = actor_action.detach().squeeze().numpy()
        x_next_pred_np = self.model.predict(x_np, u_np)
        x_next_pred_torch = torch.FloatTensor(x_next_pred_np).unsqueeze(0)
        x_next_pred_torch.requires_grad_(True)

        # 3️⃣ Critic evaluation
        V_next = self.critic(x_next_pred_torch)
        actor_loss_value = -V_next.item()  # <-- scalar for plotting/logging

        # 4️⃣ Compute ∂V/∂x_{t+1}
        V_next.backward()
        dVdx = x_next_pred_torch.grad.detach().numpy().squeeze()  # shape: (obs_dim,)

        # 5️⃣ Get model Jacobian G_t and compute manual gradient
        _, G_t = self.model.get_jacobians()  # F_t, G_t
        dLdu = -torch.FloatTensor(dVdx @ G_t)  # shape: (act_dim,)

        # 6️⃣ Apply manual gradient to actor
        self.actor.zero_grad()
        actor_action.backward(dLdu.unsqueeze(0))
        self.actor_optimizer.step()

        # ------- Update memory -------
        self.prev_obs = obs_torch.detach()
        self.prev_reward = reward_torch.detach()

        # ------- Compute model error -------
        model_error = (next_obs_torch - torch.FloatTensor(x_next_pred_np).unsqueeze(0)).abs().mean().item()

        # ------- Collect metrics -------
        actor_params = [p.data.cpu().numpy().flatten() for p in self.actor.parameters()]
        critic_params = [p.data.cpu().numpy().flatten() for p in self.critic.parameters()]
        actor_grads = [p.grad.data.cpu().numpy().flatten() for p in self.actor.parameters() if p.grad is not None]
        critic_grads = [p.grad.data.cpu().numpy().flatten() for p in self.critic.parameters() if p.grad is not None]

        return {
            'critic_error': td_error.abs().item(),
            'critic_prediction': value_pred.detach().squeeze(0).numpy(),
            'critic_target': target_value.detach().squeeze(0).numpy(),
            'model_error': model_error,
            'model_prediction': x_next_pred_np,
            'true_state': next_obs_torch.detach().squeeze(0).numpy(),
            'losses': {
                'actor_loss': float(actor_loss_value),
                'critic_loss': critic_loss.item(),
            },
            'weights_norm': {
                'actor': np.linalg.norm(np.concatenate(actor_params)),
                'critic': np.linalg.norm(np.concatenate(critic_params)),
            },
            'weights_update_norm': {
                'actor': self._compute_norm(actor_grads),
                'critic': self._compute_norm(critic_grads),
            },
            'gradients_norm': {
                'actor': self._compute_norm(actor_grads),
                'critic': self._compute_norm(critic_grads),
            }
        }

