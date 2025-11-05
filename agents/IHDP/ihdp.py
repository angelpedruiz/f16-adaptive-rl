'''
(IHDP) Incremental Heuristic Dynamic Programming Agent. Uses (RLS) for the model.
'''
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym



class Actor(nn.Module):
    ''' Actor Network for HDP outputs [-1,1] '''
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list):
        super().__init__()
        
        layers = []
        prev_size = obs_dim

        # Hidden layers
        for hidden_size in hidden_sizes:
            linear = nn.Linear(prev_size, hidden_size)
            # Smaller initialization for stability
            nn.init.normal_(linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(linear.bias)
            layers.append(linear) 
            layers.append(nn.Tanh())
            prev_size = hidden_size

        # Output layer with small uniform initialization
        output_layer = nn.Linear(prev_size, act_dim)
        nn.init.uniform_(output_layer.weight, -3e-3, 3e-3)
        nn.init.zeros_(output_layer.bias)  # Zero bias initialization
        layers.append(output_layer)
        layers.append(nn.Tanh())  # output in [-1,1]

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
            linear = nn.Linear(prev_size, hidden_size)
            nn.init.normal_(linear.weight, mean=0.0, std=0.001)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.Tanh())
            prev_size = hidden_size

        # Output layer with small initialization
        output_layer = nn.Linear(prev_size, 1)
        nn.init.uniform_(output_layer.weight, -3e-3, 3e-3)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
    def clip_pre_tanh_weights(self, clip_value=1.0):
        modules = list(self.model.children())
        for i, layer in enumerate(modules[:-1]):  # skip last layer
            if isinstance(layer, nn.Linear) and isinstance(modules[i + 1], nn.Tanh):
                with torch.no_grad():
                    layer.weight.clamp_(-clip_value, clip_value)
                    layer.bias.clamp_(-clip_value, clip_value)





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
            u: Control input (act_dim,) - in actual action space scale
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
            u: Control input (act_dim,) - in actual action space scale

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
    def __init__(self, obs_space: gym.Space, act_space: gym.Space, gamma: float, forgetting_factor: float, initial_covariance: float, hidden_sizes: dict[str, list[int]], learning_rates: dict[str, float], weight_limit: float = 30.0):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Spaces
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]
        
        # Hyperparameters
        self.gamma = gamma
        self.weight_limit = weight_limit

        # Initialize Actor, Critic, and Model networks
        self.actor = Actor(self.obs_dim, self.act_dim, hidden_sizes['actor'])
        self.critic = Critic(self.obs_dim, hidden_sizes['critic'])

        self.model = RLSModel(self.obs_dim, self.act_dim, forgetting_factor=forgetting_factor, delta=initial_covariance)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rates['actor'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rates['critic'])
        
        # Memory
        self.prev_obs = None
        self.prev_reward = None
        self.step = 0

    def _debug(self, step: int, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray,
               value_pred: torch.Tensor, target_value: torch.Tensor, td_error: torch.Tensor,
               critic_weight_before: torch.Tensor, critic_weight_after: torch.Tensor,
               actor_action_scaled: torch.Tensor, x_next_pred_np: np.ndarray, dVdx: np.ndarray,
               dVdx_G: np.ndarray, dLdu: torch.Tensor, G_t: np.ndarray, current_action: float,
               lr_actor: float, dldu_val: float, predicted_action_change: float, predicted_new_action: float,
               new_action: float):
        """
        Debug printing for the update step. Only called when step % 10 == 0.
        """
        if step == 0:
            print("\n=== Critic Architecture ===")
            for i, layer in enumerate(self.critic.model):
                print(f"Layer {i}: {layer}")
            print("=" * 40 + "\n")

        print(f"\n{'='*80}")
        print(f"STEP {step}")
        print(f"{'='*80}")

        # Environment state
        print("\n--- ENVIRONMENT STATE ---")
        print(f"State: {obs}")
        print(f"  cart_pos={obs[0]:.4f}, cart_vel={obs[1]:.4f}, angle={obs[2]:.4f}, angular_vel={obs[3]:.4f}")
        print(f"Action taken (from env): {action.item():.4f} N")
        print(f"Reward: {reward:.4f}")
        print(f"Next state: {next_obs}")

        # Critic update
        print("\n--- CRITIC UPDATE ---")
        print(f"V_pred={value_pred.item():.4f}, V_target={target_value.item():.4f}, TD_error={td_error.item():.4f}")
        print(f"Critic weight change: {(critic_weight_after - critic_weight_before).abs().max().item():.6f}")

        # Actor forward pass
        print("\n--- ACTOR FORWARD PASS ---")
        print(f"Actor output (what it wants to do): {actor_action_scaled.detach().squeeze().item():.4f} N")
        print(f"(Compare to action taken: {action.item():.4f} N)")

        # Model prediction
        print("\n--- MODEL PREDICTION ---")
        print(f"G matrix (control sensitivity):\n{G_t}")
        print(f"Model predicts next state: {x_next_pred_np}")
        print(f"True next state:           {next_obs}")
        model_error = np.linalg.norm(x_next_pred_np - next_obs)
        print(f"Model prediction error: {model_error:.6f}")

        # Critic evaluation of predicted state
        V_next_val = self.critic(torch.FloatTensor(x_next_pred_np).unsqueeze(0)).item()
        actor_loss_value = -V_next_val
        print("\n--- CRITIC EVALUATION OF PREDICTED STATE ---")
        print(f"V(predicted_next_state) = {V_next_val:.4f}")
        print(f"Actor loss (negative value) = {actor_loss_value:.4f}")

        # Critic gradient
        print("\n--- CRITIC GRADIENT (dV/dx) ---")
        print(f"dV/dx = {dVdx}")
        print(f"  dV/d(cart_pos) = {dVdx[0]:.6f} {'(+: right is better, -: left is better)' if abs(dVdx[0]) > 1e-6 else '(~0: neutral)'}")
        print(f"  dV/d(cart_vel) = {dVdx[1]:.6f}")
        print(f"  dV/d(angle)    = {dVdx[2]:.6f} {'(+: more upright is better, -: more fallen is better)' if abs(dVdx[2]) > 1e-6 else '(~0: neutral)'}")
        print(f"  dV/d(ang_vel)  = {dVdx[3]:.6f}")
        dVdx_norm = np.linalg.norm(dVdx)
        print(f"||dV/dx|| = {dVdx_norm:.6f}")

        if dVdx_norm < 1e-6:
            print("⚠️  WARNING: Critic gradient vanishing!")
        else:
            print("✓ Gradient magnitude is healthy")

        # Actor gradient computation
        print("\n--- ACTOR GRADIENT COMPUTATION ---")
        print(f"dV/dx @ G = {dVdx_G} (this is ∂V/∂u)")
        dvdu_val = dVdx_G.item() if dVdx_G.size == 1 else dVdx_G[0]
        dldu_val_actual = dLdu.item() if dLdu.numel() == 1 else dLdu[0].item()
        print(f"  Interpretation: If action increases by 1N, V changes by {dvdu_val:.6f}")
        print(f"dL/du = -{dvdu_val:.6f} = {dldu_val_actual:.6f}")
        print(f"  Interpretation: Gradient says action should {'INCREASE' if dldu_val_actual > 0 else 'DECREASE'}")
        print(f"||dL/du|| = {torch.norm(dLdu).item():.6f}")

        # Gradient step prediction
        print("\n--- GRADIENT STEP PREDICTION ---")
        print(f"Learning rate: {lr_actor}")
        print(f"Current action output: {current_action:.4f} N")
        print(f"Gradient: {dldu_val:.6f}")
        print(f"Predicted action change: {predicted_action_change:.6f} N")
        print(f"Predicted new action: {predicted_new_action:.4f} N")

        # Sanity checks
        if abs(predicted_new_action) > 9.0:
            print("⚠️  WARNING: Actor will saturate at action limits!")
        if abs(dldu_val) > 10.0:
            print("⚠️  WARNING: Very large gradient - might cause instability!")

        # Actual parameter gradients
        print("\n--- ACTUAL PARAMETER GRADIENTS (before clipping) ---")
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad_norm={param.grad.norm().item():.6f}, grad_mean={param.grad.mean().item():.6f}")

        # After actor update
        print("\n--- AFTER ACTOR UPDATE ---")
        print(f"New actor output: {new_action:.4f} N (was {current_action:.4f} N)")
        print(f"Actual change: {new_action - current_action:.6f} N")
        print(f"{'='*80}\n")
                


    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from actor network - [-1,1]"""
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs)
        action = action.squeeze(0).numpy()
        return action # [-1,1]

    @staticmethod
    def _compute_norm(grad_list):
        """Compute L2 norm of gradients, handling empty lists."""
        if not grad_list:
            return 0.0
        return np.linalg.norm(np.concatenate(grad_list))

    def _diagnose_critic_activations(self, obs: torch.Tensor) -> dict:
        """
        Diagnose critic network layer activations to detect saturation and gradient flow issues.

        Returns dict with activation statistics for each layer.
        """
        stats = {}

        # Forward pass through critic layers, tracking activations
        x = obs
        for i, layer in enumerate(self.critic.model):
            x = layer(x)

            # Get statistics for this layer
            x_flat = x.detach().cpu().numpy().flatten()

            layer_type = layer.__class__.__name__
            stats[f'layer_{i}'] = {
                'type': layer_type,
                'mean': float(np.mean(x_flat)),
                'std': float(np.std(x_flat)),
                'min': float(np.min(x_flat)),
                'max': float(np.max(x_flat)),
            }

            # For Tanh, check saturation (values near ±1)
            if layer_type == 'Tanh':
                saturation_count = np.sum(np.abs(x_flat) > 0.99)
                saturation_pct = 100.0 * saturation_count / len(x_flat)
                stats[f'layer_{i}']['saturation_pct'] = saturation_pct
                stats[f'layer_{i}']['saturated'] = saturation_pct > 20  # Flag if >20% saturated

        return stats

    def _verify_critic_gradient_finite_diff(self, x_torch: torch.Tensor, dVdx_autograd: np.ndarray, epsilon: float = 1e-5) -> dict:
        """
        Verify autograd gradients using finite-difference approximation.

        Compares analytical gradients (from autograd) with numerical gradients (from finite differences).
        This helps detect issues with gradient computation.

        Args:
            x_torch: Input state tensor (requires_grad=True)
            dVdx_autograd: Autograd-computed gradient array
            epsilon: Small perturbation for finite differences

        Returns:
            Dictionary with comparison statistics
        """
        x_np = x_torch.detach().cpu().numpy().flatten()
        state_dim = len(x_np)

        dVdx_fd = np.zeros(state_dim)

        # Compute finite-difference gradient for each dimension
        for i in range(state_dim):
            # Forward perturbation
            x_plus = x_np.copy()
            x_plus[i] += epsilon
            x_plus_torch = torch.FloatTensor(x_plus).unsqueeze(0)
            V_plus = self.critic(x_plus_torch).item()

            # Backward perturbation
            x_minus = x_np.copy()
            x_minus[i] -= epsilon
            x_minus_torch = torch.FloatTensor(x_minus).unsqueeze(0)
            V_minus = self.critic(x_minus_torch).item()

            # Central difference: (f(x+eps) - f(x-eps)) / (2*eps)
            dVdx_fd[i] = (V_plus - V_minus) / (2.0 * epsilon)

        # Compare autograd vs finite-difference
        diff = np.abs(dVdx_autograd - dVdx_fd)
        rel_error = diff / (np.abs(dVdx_fd) + 1e-10)  # Avoid division by zero

        return {
            'dVdx_autograd': dVdx_autograd,
            'dVdx_fd': dVdx_fd,
            'abs_diff': diff,
            'rel_error': rel_error,
            'max_abs_diff': float(np.max(diff)),
            'max_rel_error': float(np.max(rel_error[np.isfinite(rel_error)])),
        }

    def _diagnose_critic_gradients(self, obs: torch.Tensor, target_value: torch.Tensor) -> dict:
        """
        Diagnose gradient flow through critic network layers.

        Returns dict with gradient statistics for each parameter.
        """
        # Forward pass
        value_pred = self.critic(obs)
        loss = 0.5 * (target_value - value_pred).pow(2).mean()

        # Backward pass
        self.critic_optimizer.zero_grad()
        loss.backward()

        grad_stats = {}
        for i, (name, param) in enumerate(self.critic.named_parameters()):
            if param.grad is not None:
                grad_np = param.grad.data.cpu().numpy()
                grad_stats[name] = {
                    'grad_norm': float(np.linalg.norm(grad_np)),
                    'grad_mean': float(np.mean(np.abs(grad_np))),
                    'grad_max': float(np.max(np.abs(grad_np))),
                    'param_norm': float(np.linalg.norm(param.data.cpu().numpy())),
                }

        return grad_stats

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, next_obs: np.ndarray) -> dict:
        """
        Update Actor, Critic, and Model networks based on a single transition tuple.
        """

        # --- Convert inputs to torch tensors ---
        obs_torch = torch.FloatTensor(obs).unsqueeze(0)
        action_torch = torch.FloatTensor(action).unsqueeze(0)
        reward_torch = torch.FloatTensor([reward]).unsqueeze(0)
        next_obs_torch = torch.FloatTensor(next_obs).unsqueeze(0)
        
        # ---- Check critic activation stats on first step ----
        if self.step == 0:
            print("\n=== Critic Architecture ===")
            for i, layer in enumerate(self.critic.model):
                print(f"Layer {i}: {layer}")
            print("=" * 40 + "\n")

        # ------- Update Model (with scaled action from environment) -------
        self.model.update(obs_torch.detach().numpy().squeeze(),
                        action_torch.detach().numpy().squeeze(),
                        next_obs_torch.detach().numpy().squeeze())

        # ------- Update Critic (TD(0)) -------
        critic_weight_before = list(self.critic.parameters())[0].data.clone()

        with torch.no_grad():
            target_value = reward_torch + self.gamma * self.critic(next_obs_torch)

        value_pred = self.critic(obs_torch)
        td_error = target_value - value_pred
        critic_loss = 0.5 * td_error.pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        self.critic.clip_pre_tanh_weights(clip_value=self.weight_limit)

        critic_weight_after = list(self.critic.parameters())[0].data

        # ------- Update Actor (manual gradient via RLS Jacobian) -------
        # 1️⃣ Forward pass through actor
        actor_action_scaled = self.actor(obs_torch)
        actor_action_scaled.retain_grad()

        # 2️⃣ Predict next state with actor's scaled action
        x_np = obs_torch.detach().squeeze().numpy()
        u_np = actor_action_scaled.detach().squeeze().numpy()

        x_next_pred_np = self.model.predict(x_np, u_np)
        x_next_pred_torch = torch.FloatTensor(x_next_pred_np).unsqueeze(0)
        x_next_pred_torch.requires_grad_(True)

        # 3️⃣ Critic evaluation on predicted next state
        V_next = self.critic(x_next_pred_torch)
        actor_loss_value = -V_next.item()

        # 4️⃣ Compute ∂V/∂x_{t+1}
        V_next.backward()
        dVdx = x_next_pred_torch.grad.detach().numpy().squeeze()

        # 5️⃣ Get model Jacobian G_t
        _, G_t = self.model.get_jacobians()

        # 6️⃣ Compute gradient: ∂L/∂u = -∂V/∂x * ∂x/∂u
        dVdx_G = dVdx @ G_t  # This is ∂V/∂u (how V changes with action)
        dLdu = -torch.FloatTensor(dVdx_G)  # Negative because we want to maximize V

        # Prepare debug variables (needed for _debug method)
        current_action = actor_action_scaled.detach().squeeze().item()
        lr_actor = self.actor_optimizer.param_groups[0]['lr']
        dldu_val = dLdu.item() if dLdu.numel() == 1 else dLdu[0].item()
        predicted_action_change = -lr_actor * dldu_val
        predicted_new_action = current_action + predicted_action_change

        # 7️⃣ Apply manual gradient to actor
        self.actor.zero_grad()
        actor_action_scaled.backward(dLdu.unsqueeze(0))

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        new_action = self.actor(obs_torch).detach().squeeze().item()

        # Call debug method (only prints when step % 10 == 0)
        if self.step % 10 == 0:
            self._debug(
                step=self.step,
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                value_pred=value_pred,
                target_value=target_value,
                td_error=td_error,
                critic_weight_before=critic_weight_before,
                critic_weight_after=critic_weight_after,
                actor_action_scaled=actor_action_scaled,
                x_next_pred_np=x_next_pred_np,
                dVdx=dVdx,
                dVdx_G=dVdx_G,
                dLdu=dLdu,
                G_t=G_t,
                current_action=current_action,
                lr_actor=lr_actor,
                dldu_val=dldu_val,
                predicted_action_change=predicted_action_change,
                predicted_new_action=predicted_new_action,
                new_action=new_action
            )

            # --- Run diagnostics ---
            print("\n=== CRITIC ACTIVATION DIAGNOSTICS ===")
            activation_stats = self._diagnose_critic_activations(obs_torch)
            for layer_name, stats in activation_stats.items():
                print(f"\n{layer_name} ({stats['type']}):")
                print(f"  Output range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                if 'saturation_pct' in stats:
                    sat_pct = stats['saturation_pct']
                    print(f"  Tanh Saturation: {sat_pct:.1f}%", end="")
                    if stats['saturated']:
                        print(" ⚠️  HIGH SATURATION!")
                    else:
                        print()

            print("\n=== CRITIC GRADIENT DIAGNOSTICS ===")
            grad_stats = self._diagnose_critic_gradients(obs_torch, target_value)
            for param_name, stats in grad_stats.items():
                print(f"{param_name}:")
                print(f"  Grad norm: {stats['grad_norm']:.8f}, Grad mean: {stats['grad_mean']:.8f}, Grad max: {stats['grad_max']:.8f}")
                print(f"  Param norm: {stats['param_norm']:.6f}")
                if stats['grad_norm'] < 1e-6:
                    print("  ⚠️  VANISHING GRADIENT!")

            print("\n=== FINITE-DIFFERENCE GRADIENT VERIFICATION ===")
            print("Comparing autograd dVdx with finite-difference approximation...")
            fd_verify = self._verify_critic_gradient_finite_diff(x_next_pred_torch, dVdx)
            state_labels = ['cart_pos', 'cart_vel', 'angle', 'ang_vel']
            print(f"\n{'State':<15} {'Autograd':<15} {'Finite-Diff':<15} {'Abs Diff':<15} {'Rel Error':<15}")
            print("-" * 75)
            for i in range(len(dVdx)):
                label = state_labels[i] if i < len(state_labels) else f'state_{i}'
                ag = fd_verify['dVdx_autograd'][i]
                fd = fd_verify['dVdx_fd'][i]
                ad = fd_verify['abs_diff'][i]
                re = fd_verify['rel_error'][i]
                print(f"{label:<15} {ag:<15.8f} {fd:<15.8f} {ad:<15.8e} {re:<15.4f}")
            print(f"\nMax absolute difference: {fd_verify['max_abs_diff']:.8e}")
            print(f"Max relative error: {fd_verify['max_rel_error']:.4f}")
            if fd_verify['max_abs_diff'] > 1e-3:
                print("⚠️  Large discrepancy between autograd and finite-diff! Check gradient computation.")

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

        # Collect separate weights for each layer (use .copy() to ensure independent snapshots)
        actor_weights_by_layer = {}
        for i, param in enumerate(self.actor.parameters()):
            actor_weights_by_layer[f'layer_{i}'] = param.data.cpu().numpy().copy()

        critic_weights_by_layer = {}
        for i, param in enumerate(self.critic.parameters()):
            critic_weights_by_layer[f'layer_{i}'] = param.data.cpu().numpy().copy()

        self.step += 1

        return {
            'critic_error': td_error.abs().item(),
            'critic_prediction': value_pred.detach().squeeze(0).numpy(),
            'critic_target': target_value.detach().squeeze(0).numpy(),
            'model_error': model_error,
            'model_prediction': x_next_pred_np,
            'true_state': next_obs_torch.detach().squeeze(0).numpy(),
            'G_norm': float(np.linalg.norm(G_t)),
            'action_gradient_norm': float(np.linalg.norm(dLdu.numpy())),
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
            },
            'actor_weights': actor_weights_by_layer,
            'critic_weights': critic_weights_by_layer,
            'dVdx': dVdx,
        }