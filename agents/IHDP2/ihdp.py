from torch import nn
import torch
import numpy as np
import gymnasium as gym

class Actor(nn.Module):
    '''Actor Network with decaying learning rate.'''
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int], lr: float, lr_decay: float, lr_min: float):
        """Initialize the Actor network."""
        super().__init__()
        layers = []
        for h in hidden_sizes:
            layers += [nn.Linear(obs_dim, h), nn.Tanh()]
            obs_dim = h
        layers += [nn.Linear(obs_dim, act_dim), nn.Tanh()]  # final tanh for bounded actions
        self.net = nn.Sequential(*layers)

        # Learning rate and decay parameters
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_min = lr_min

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, state):
        '''Define the forward pass of the Actor network.'''
        return self.net(state) # Output action in range [-1, 1] due to final Tanh

    def update(self, xt: torch.Tensor, Vt1: torch.Tensor, dVt1dxt1: torch.Tensor, G: np.ndarray):
        """
        Perform one online actor update using critic information.
        xt: current state (batch_size x state_dim)
        Vt1: critic evaluation of next state (batch_size x 1)
        dVt1dxt1: gradient of critic w.r.t next state (batch_size x state_dim)
        G: input distribution matrix (state_dim x action_dim)
        """
        
        chain_rule = (Vt1 * torch.matmul(G.T, dVt1dxt1)).mean()
        self.optimizer.zero_grad()
        
        loss = 0.5 * chain_rule**2
        loss.backward()
        self.optimizer.step()
        
        # Decay learning rate
        self._apply_alpha_decay()
    
    def _exploration(self, action, noise_scale=0.1):
        noise = noise_scale * torch.randn_like(action)
        return torch.clamp(action + noise, -1, 1)
    
    def _apply_alpha_decay(self):
        '''Apply alpha decay to the Actor network.'''
        for g in self.optimizer.param_groups:
            g['lr'] = max(self.lr_min, g['lr'] * self.lr_decay)
        
    

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_layers: list[int], gamma: float, lr: float, lr_decay: float = 0.99, lr_min: float = 1e-5):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_layers = hidden_layers
        self.gamma = gamma
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_min = lr_min

        # Memory
        self.xt_1 = None
        self.rt_1 = None
        
        # Construct network
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    def update(self, xt: torch.Tensor, rt: float, terminated: bool, xt1_est: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # FIRST: Update parameters if we have previous state
        if self.xt_1 is not None:
            V_prev = self.forward(self.xt_1)
            Vt = self.forward(xt)
            
            # TD target: r_{t-1} + gamma * V(x_t)
            target = self.rt_1 + (0 if terminated else self.gamma * Vt.detach())
            
            # TD error
            error = target - V_prev
            loss = 0.5 * error**2
            
            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # THEN: Compute Vt1 and gradient with updated network
        xt1_est = xt1_est.detach().requires_grad_(True)
        
        Vt1 = self.forward(xt1_est)
        dVt1dxt1 = torch.autograd.grad(Vt1, xt1_est, create_graph=False)[0]
        
        # Decay learning rate
        self._decay_lr()
        
        # Update memory for next iteration
        self.xt_1 = xt.detach()
        self.rt_1 = rt

        return Vt1.detach(), dVt1dxt1.detach()
    
    def _decay_lr(self):
        '''Apply learning rate decay to the Critic network.'''
        for g in self.optimizer.param_groups:
            g['lr'] = max(self.lr_min, g['lr'] * self.lr_decay)


class IncrementalModel:
    ''' May need to include limits in the future.'''
    def __init__(self, obs_dim: int, act_dim: int, window_size: int = None):
        """
        Incremental Model with Least Squares.
        obs_dim : number of state variables
        act_dim : number of control inputs
        L       : memory length for LS (default: 2*(obs_dim + act_dim))
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.L = window_size if window_size is not None else 2 * (obs_dim + act_dim)
        
        # Circular buffer for deltas
        self.delta_x_store = np.zeros((self.L, obs_dim))
        self.delta_u_store = np.zeros((self.L, act_dim))
        
        # State tracking
        self.x_prev = None
        self.u_prev = None
        self.x_curr = None
        self.u_curr = None
        
        # Model matrices
        self.F = np.zeros((obs_dim, obs_dim))
        self.G = np.zeros((obs_dim, act_dim))
        
        self.time_step = 0

    def predict(self) -> np.ndarray:
        """Predict next state using current incremental model."""
        if self.x_curr is None or self.x_prev is None:
            return self.x_curr
        
        delta_x = self.x_curr - self.x_prev
        delta_u = self.u_curr - self.u_prev
        delta_x_next = self.F @ delta_x + self.G @ delta_u
        
        return self.x_curr + delta_x_next
    
    def update(self, x: np.ndarray, u: np.ndarray, next_x: np.ndarray):
        """Update model with new observation."""
        x = np.asarray(x).reshape(-1, 1)
        u = np.asarray(u).reshape(-1, 1)
        next_x = np.asarray(next_x).reshape(-1, 1)
        
        # Initialize on first step
        if self.time_step == 0:
            self.x_prev = x
            self.u_prev = u
            self.x_curr = x
            self.u_curr = u
            self.time_step += 1
            return
        
        # Compute deltas
        delta_x = x - self.x_prev
        delta_u = u - self.u_prev
        delta_x_next = next_x - x
        
        # Store in circular buffer
        idx = (self.time_step - 1) % self.L
        self.delta_x_store[idx] = delta_x.ravel()
        self.delta_u_store[idx] = delta_u.ravel()
        
        # Perform LS if enough data
        if self.time_step > 1:
            n_samples = min(self.time_step - 1, self.L)
            A = np.hstack([self.delta_x_store[:n_samples], 
                          self.delta_u_store[:n_samples]])
            b = self.delta_x_store[1:n_samples+1]
            A = np.hstack([self.delta_x_store[:n_samples], self.delta_u_store[:n_samples]])

            
            # Solve LS
            params = np.linalg.lstsq(A, b, rcond=None)[0]
            self.F = params[:self.obs_dim].T
            self.G = params[self.obs_dim:].T
        
        # Update state
        self.x_prev = x
        self.u_prev = u
        self.x_curr = next_x
        self.u_curr = u
        self.time_step += 1


class IHDPAgent():
    def __init__(self, obs_space: gym.spaces.Box, act_space: gym.spaces.Box, parameters: dict):
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))
        self.gamma = parameters.get('gamma', 0.99)

        # actor / critic params
        actor_hidden = parameters.get('actor_hidden', [64, 64])
        critic_hidden = parameters.get('critic_hidden', [64, 64])
        actor_lr = parameters.get('learning_rate', 1e-3)
        actor_lr_decay = parameters.get('actor_lr_decay', 0.995)
        actor_lr_min = parameters.get('actor_lr_min', 1e-5)
        critic_lr = parameters.get('critic_learning_rate', 1e-3)
        critic_lr_decay = parameters.get('critic_lr_decay', 0.995)
        critic_lr_min = parameters.get('critic_lr_min', 1e-6)

        self.actor = Actor(obs_dim=obs_dim, act_dim=act_dim,
                           hidden_sizes=actor_hidden, lr=actor_lr,
                           lr_decay=actor_lr_decay, lr_min=actor_lr_min)

        self.critic = Critic(obs_dim=obs_dim, hidden_layers=critic_hidden,
                             gamma=self.gamma, lr=critic_lr,
                             lr_decay=critic_lr_decay, lr_min=critic_lr_min)

        self.model = IncrementalModel(obs_dim=obs_dim, act_dim=act_dim)

    
    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        '''Get action from the agent, to return to environment.'''
        action = self.actor(obs).detach().numpy()
        return action

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, next_obs: np.ndarray):
        '''Update the agent's networks and models. main iteration loop: 1:agent.get_action(); 2:env.step(); 3:agent.update()'''
        xt = obs
        ut = action
        rt = reward
        xt1 = next_obs
            
        # Predict next state, get G matrix and update model.
        xt1est = self.model.predict()
        G = self.model.G
        self.model.update(xt, ut, xt1)
        
        # Update critic
        Vt1, dVt1dxt1 = self.critic.update(xt, rt, terminated, xt1est)
        G_torch = torch.tensor(G, dtype=dVt1dxt1.dtype, device=dVt1dxt1.device)
        
        # Update actor
        self.actor.update(xt, Vt1, dVt1dxt1, G_torch)