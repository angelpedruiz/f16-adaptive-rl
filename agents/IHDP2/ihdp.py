from torch import nn
import torch
import numpy as np
import gimnasium as gym

class Actor(nn.Module):
    def __init__(self):
        '''Initialize the Actor network.'''
        pass

    def forward(self, state):
        '''Define the forward pass of the Actor network.'''
        pass
    
    def update(self):
        '''Define the update mechanism for the Actor network.'''
        pass
    
    def _exploration(self):
        '''Define the exploration strategy for the Actor network.'''
        pass  
    
    def _apply_alpha_decay(self):
        '''Apply alpha decay to the Actor network.'''
        pass
    

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_layers: list[int], gamma: float, lr: float):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_layers = hidden_layers
        self.gamma = gamma
        self.lr = lr
        
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
        dVt1dxt1 = torch.autograd.grad(Vt1, xt1_est, retain_graph=True, create_graph=False)[0]
        
        # Update memory for next iteration
        self.xt_1 = xt.detach()
        self.rt_1 = rt

        return Vt1.detach(), dVt1dxt1.detach()


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
            b = np.roll(self.delta_x_store, -1, axis=0)[:n_samples]
            
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
        self.gamma = parameters.get('gamma')
        self.actor_lr = parameters.get('learning_rate')
        self.critic_lr = parameters.get('critic_learning_rate')
        

        self.actor = Actor()
        self.critic = Critic()
        self.model = IncrementalModel()
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        '''Get action from the agent, to return to environment.'''
        return self.actor(obs)

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, next_obs: np.ndarray):
        '''Update the agent's networks and models. main iteration loop: 1:agent.get_action(); 2:env.step(); 3:agent.update()'''
        xt = obs
        ut = action
        r = reward
        xt1 = next_obs
            
        # Predict next state, get G matrix and update model.
        xt1est = self.model.predict()
        G = self.model.G
        self.model.update(xt, ut, xt1)
        
        # Update critic
        Vt1, dVt1dxt1 = self.critic.update(xt, ut, r, xt1, terminated, xt1est)
        
        # Update actor
        self.actor.update(xt, Vt1, dVt1dxt1, G)