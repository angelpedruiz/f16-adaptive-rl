from torch import nn
import torch
import numpy as np
import gymnasium as gym

class Actor(nn.Module):
    '''Actor Network with decaying learning rate.'''
    def __init__(self, obs_space: gym.spaces.Box, act_space: gym.spaces.Box, hidden_sizes: list[int], lr: float, lr_decay: float, lr_min: float):
        """Initialize the Actor network."""
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]

        super().__init__()
        layers = []
        input_dim = self.obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(input_dim, h), nn.Tanh()]
            input_dim = h
        layers += [nn.Linear(input_dim, self.act_dim), nn.Tanh()]  # final tanh for bounded actions
        self.net = nn.Sequential(*layers)

        # Learning rate and decay parameters
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_min = lr_min

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        '''Define the forward pass of the Actor network.'''
        return self.net(obs) # [-1,1]
    
    def update(self, xt: torch.Tensor, Vt1: torch.Tensor, dVt1dxt1: torch.Tensor, G: np.ndarray):
        """
        Perform one online actor update using critic information.
        xt: current state (batch_size x state_dim)
        Vt1: critic evaluation of next state (batch_size x 1)
        dVt1dxt1: gradient of critic w.r.t next state (batch_size x state_dim)
        G: input distribution matrix (state_dim x action_dim)
        """

        # Get actor output for this state
        ut = self.forward(xt)  # Actor output (batch x action_dim)

        # Compute dV/du = dV/dx @ dx/du = dVt1dxt1 @ G
        dVdu = torch.matmul(dVt1dxt1, G)

        # Actor loss: minimize -(dV/du * u), i.e., maximize value increase through chosen action
        # The actor learns to choose actions that increase the value
        loss = (-dVdu * ut).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay learning rate
        self._apply_alpha_decay()
    
    def exploration(self, action, noise_scale=0.1):
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
            layers.append(nn.Tanh())
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
        self.delta_x_next_store = np.zeros((self.L, obs_dim))
        
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
        x = np.asarray(x).flatten()
        u = np.asarray(u).flatten()
        next_x = np.asarray(next_x).flatten()

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
        self.delta_x_store[idx] = delta_x
        self.delta_u_store[idx] = delta_u
        self.delta_x_next_store[idx] = delta_x_next

        # Perform LS if enough data
        if self.time_step > 1:
            n_samples = min(self.time_step - 1, self.L)
            A = np.hstack([self.delta_x_store[:n_samples],
                          self.delta_u_store[:n_samples]])
            b = self.delta_x_next_store[:n_samples]

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
        """
        Initialize IHDP Agent with spaces and parameters.

        Args:
            obs_space (gym.spaces.Box): Observation space
            act_space (gym.spaces.Box): Action space
            parameters (dict): Configuration dictionary with keys:
                - gamma (float, default=0.99): Discount factor
                - actor_hidden (list[int], default=[64,64]): Actor hidden layer sizes
                - critic_hidden (list[int], default=[64,64]): Critic hidden layer sizes
                - learning_rate (float, default=1e-3): Actor learning rate
                - actor_lr_decay (float, default=0.995): Actor LR decay
                - actor_lr_min (float, default=1e-5): Actor minimum LR
                - critic_learning_rate (float, default=1e-3): Critic learning rate
                - critic_lr_decay (float, default=0.995): Critic LR decay
                - critic_lr_min (float, default=1e-6): Critic minimum LR
                - exploration_noise_scale (float, default=0.1): Action exploration noise
                - model_window_size (int, default=None): Incremental model buffer size
        """
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))
        self.act_low = torch.tensor(act_space.low, dtype=torch.float32)
        self.act_high = torch.tensor(act_space.high, dtype=torch.float32)
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

        # Exploration and model params
        self.exploration_noise_scale = parameters.get('exploration_noise_scale', 0.1)
        model_window_size = parameters.get('model_window_size', None)

        self.actor = Actor(obs_space=obs_space, act_space=act_space,
                           hidden_sizes=actor_hidden, lr=actor_lr,
                           lr_decay=actor_lr_decay, lr_min=actor_lr_min)

        self.critic = Critic(obs_dim=obs_dim, hidden_layers=critic_hidden,
                             gamma=self.gamma, lr=critic_lr,
                             lr_decay=critic_lr_decay, lr_min=critic_lr_min)

        self.model = IncrementalModel(obs_dim=obs_dim, act_dim=act_dim, window_size=model_window_size)
        self.step = 0



    def get_action(self, obs: np.ndarray) -> np.ndarray:
        '''Get action from the agent, to return to environment.'''
        obs_torch = torch.tensor(obs, dtype=torch.float32)
        if obs_torch.dim() == 1:
            obs_torch = obs_torch.unsqueeze(0)  # Add batch dimension
        action = self.actor(obs_torch).detach()

        # Apply exploration
        action_explored = self.actor.exploration(action, noise_scale=self.exploration_noise_scale)

        # Scale action to environment bounds
        scaled_action = action_explored.detach().numpy() * (self.act_high - self.act_low).numpy() / 2 + (self.act_high + self.act_low).numpy() / 2
        return scaled_action.squeeze(0) if scaled_action.shape[0] == 1 else scaled_action

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, next_obs: np.ndarray) -> dict:
        '''
        Update the agent's networks and models. Main iteration loop: 1:agent.get_action(); 2:env.step(); 3:agent.update()

        Returns:
            dict: Metrics for monitoring and plotting
        '''
        # Only learn after first step
        
        # Convert inputs to torch tensors
        xt = torch.tensor(obs, dtype=torch.float32)
        if xt.dim() == 1:
            xt = xt.unsqueeze(0)
        ut = action
        rt = torch.tensor(reward, dtype=torch.float32)
        xt1 = torch.tensor(next_obs, dtype=torch.float32)
        if xt1.dim() == 1:
            xt1 = xt1.unsqueeze(0)

        # Predict next state, get G matrix and update model
        xt1est = self.model.predict()
        G = self.model.G
        self.model.update(xt.detach().numpy().squeeze(), ut, xt1.detach().numpy().squeeze())

        # On first step, skip learning (model has no data yet)
        if self.step == 0:
            # Initialize critic memory for next step
            self.critic.xt_1 = xt.detach()
            self.critic.rt_1 = rt
            # Return placeholder metrics
            metrics = self._compute_metrics(xt, xt1, rt, torch.tensor([0.0]), torch.zeros_like(xt), G, xt1est)
        else:
            # Convert to torch tensor for critic (flatten to ensure correct shape)
            if xt1est is not None:
                xt1est_np = np.asarray(xt1est).flatten()
                xt1est_torch = torch.tensor(xt1est_np, dtype=torch.float32).unsqueeze(0)
            else:
                xt1est_torch = None

            # Update critic
            Vt1, dVt1dxt1 = self.critic.update(xt, rt, terminated, xt1est_torch)

            # Update actor
            self.actor.update(xt, Vt1, dVt1dxt1, torch.tensor(G, dtype=dVt1dxt1.dtype, device=dVt1dxt1.device))

            # Compute and return metrics
            metrics = self._compute_metrics(xt, xt1, rt, Vt1, dVt1dxt1, G, xt1est)

            # Print debug information
            self._debug_print(obs, action, reward, terminated, next_obs, Vt1, dVt1dxt1, G, xt1est, metrics)

        self.step += 1
        return metrics
    
    @staticmethod
    def _compute_grad_norm(model):
        """Compute L2 norm of all gradients in model."""
        grad_list = [p.grad.data.cpu().numpy().flatten() for p in model.parameters() if p.grad is not None]
        if not grad_list:
            return 0.0
        return float(np.linalg.norm(np.concatenate(grad_list)))

    @staticmethod
    def _compute_weight_norm(model):
        """Compute L2 norm of all weights in model."""
        weight_list = [p.data.cpu().numpy().flatten() for p in model.parameters()]
        if not weight_list:
            return 0.0
        return float(np.linalg.norm(np.concatenate(weight_list)))

    def _compute_metrics(self, xt: torch.Tensor, xt1: torch.Tensor, rt: torch.Tensor,
                        Vt1: torch.Tensor, dVt1dxt1: torch.Tensor, G: np.ndarray,
                        xt1est: np.ndarray) -> dict:
        """
        Compute all metrics for monitoring and plotting.

        Args:
            xt: Current state tensor
            xt1: Next state tensor
            rt: Reward tensor
            Vt1: Critic value at next state
            dVt1dxt1: Critic gradient w.r.t. next state
            G: Model control sensitivity matrix
            xt1est: Predicted next state

        Returns:
            dict: All metrics for logging and plotting
        """
        # Compute critic loss for logging (TD error squared)
        td_error = rt + self.gamma * Vt1 - self.critic(xt)
        critic_loss = 0.5 * (td_error ** 2).mean().item()

        # Compute model prediction error
        model_error = float(np.linalg.norm(xt1est - xt1.detach().numpy().squeeze())) if xt1est is not None else 0.0

        # Compute norms for logging
        G_norm = float(np.linalg.norm(G))
        G_torch = torch.tensor(G, dtype=dVt1dxt1.dtype, device=dVt1dxt1.device)
        dVdu_norm = float(torch.norm(dVt1dxt1 @ G_torch).item())

        # Collect weight and gradient norms
        actor_weight_norm = self._compute_weight_norm(self.actor)
        critic_weight_norm = self._compute_weight_norm(self.critic)
        actor_grad_norm = self._compute_grad_norm(self.actor)
        critic_grad_norm = self._compute_grad_norm(self.critic)

        # Ensure scalar values are returned as 1D numpy arrays for consistency
        # The critic was trained on delayed rewards: target = r_{t-1} + gamma * V(x_t)
        # Prediction was for V(x_{t-1}), which is stored in critic.xt_1

        if self.step > 0 and self.critic.xt_1 is not None:
            # Prediction: what was predicted for previous state
            critic_pred = self.critic.forward(self.critic.xt_1).detach().squeeze().numpy()
            critic_pred = np.atleast_1d(critic_pred)

            # Target: r_{t-1} + gamma * V(x_t) - the actual training target
            Vt_train = self.critic.forward(xt).detach()
            target_train = self.critic.rt_1 + self.gamma * Vt_train
            critic_target = target_train.squeeze().numpy()
            critic_target = np.atleast_1d(critic_target)
        else:
            critic_pred = np.atleast_1d(0.0)
            critic_target = np.atleast_1d(0.0)

        return {
            'critic_error': float(td_error.abs().mean().item()),
            'critic_prediction': critic_pred,
            'critic_target': critic_target,
            'model_error': model_error,
            'model_prediction': xt1est if xt1est is not None else xt1.detach().squeeze(0).numpy(),
            'true_state': xt1.detach().squeeze(0).numpy(),
            'G_norm': G_norm,
            'action_gradient_norm': dVdu_norm,
            'losses': {
                'actor_loss': float((-Vt1).mean().item()),
                'critic_loss': critic_loss,
            },
            'weights_norm': {
                'actor': actor_weight_norm,
                'critic': critic_weight_norm,
            },
            'gradients_norm': {
                'actor': actor_grad_norm,
                'critic': critic_grad_norm,
            }
        }

    def _debug_print(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool,
                     next_obs: np.ndarray, Vt1: torch.Tensor, dVt1dxt1: torch.Tensor,
                     G: np.ndarray, xt1est: np.ndarray, metrics: dict):
        """
        Print comprehensive debug information. Call once per update for monitoring.

        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            terminated: Episode termination flag
            next_obs: Next observation
            Vt1: Critic value at next state
            dVt1dxt1: Critic gradient w.r.t. next state
            G: Model control matrix
            xt1est: Model prediction of next state
            metrics: Metrics dict from _compute_metrics
        """
        if self.step % 10 != 0:
            return  # Only print every 10 steps

        print(f"\n{'='*80}")
        print(f"STEP {self.step}")
        print(f"{'='*80}")

        # Environment state
        print("\n--- ENVIRONMENT ---")
        print(f"Observation: {obs}")
        if len(obs) == 4:  # PendulumCart
            print(f"  [cart_pos={obs[0]:7.4f}, cart_vel={obs[1]:7.4f}, angle={obs[2]:7.4f}, ang_vel={obs[3]:7.4f}]")
        print(f"Action: {action.squeeze():.6f}")
        print(f"Reward: {reward:.6f}")
        print(f"Terminated: {terminated}")
        print(f"Next state: {next_obs}")

        # Critic information
        print("\n--- CRITIC ---")
        print(f"V(next_state) = {Vt1.item():.6f}")
        if self.step > 0:
            pred = metrics['critic_prediction'][0] if isinstance(metrics['critic_prediction'], np.ndarray) else metrics['critic_prediction']
            target = metrics['critic_target'][0] if isinstance(metrics['critic_target'], np.ndarray) else metrics['critic_target']
            print(f"V_pred(prev) = {pred:.6f}, V_target = {target:.6f}, TD_error = {pred - target:.6f}")
        print(f"Critic loss: {metrics['losses']['critic_loss']:.6f}")
        print(f"Critic weight norm: {metrics['weights_norm']['critic']:.6f}")
        print(f"Critic gradient norm: {metrics['gradients_norm']['critic']:.6f}")
        print(f"Critic LR: {self.critic.optimizer.param_groups[0]['lr']:.2e}")

        # Model information
        print("\n--- MODEL ---")
        F, G_mat = self.model.F, self.model.G
        print(f"F (state transition):\n{F}")
        print(f"G (control sensitivity):\n{G_mat}")
        if xt1est is not None:
            print(f"Predicted next state: {xt1est}")
            print(f"True next state:      {next_obs}")
            print(f"Model error: {metrics['model_error']:.6f}")
        print(f"G norm: {metrics['G_norm']:.6f}")

        # Actor information
        print("\n--- ACTOR ---")
        print(f"dV/du gradient norm: {metrics['action_gradient_norm']:.6f}")
        print(f"Actor loss: {metrics['losses']['actor_loss']:.6f}")
        print(f"Actor weight norm: {metrics['weights_norm']['actor']:.6f}")
        print(f"Actor gradient norm: {metrics['gradients_norm']['actor']:.6f}")
        print(f"Actor LR: {self.actor.optimizer.param_groups[0]['lr']:.2e}")

        # Learning summary
        print("\n--- LEARNING DYNAMICS ---")
        dVdu = (dVt1dxt1 @ torch.tensor(G_mat, dtype=dVt1dxt1.dtype, device=dVt1dxt1.device)).item() if G_mat.size > 0 else 0.0
        print(f"dV/du = {dVdu:.6f} (value change if action increases by 1)")
        actor_lr = self.actor.optimizer.param_groups[0]['lr']
        predicted_action_change = -actor_lr * dVdu
        print(f"Predicted action change: {predicted_action_change:.6f}")

        print(f"{'='*80}\n")
