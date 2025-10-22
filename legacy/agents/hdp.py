'''
(HDP) Heuristic Dynamic Programming Agent. Deep model.
'''
import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    ''' Actor Network for ADHDP: pi: x_t -> u_t. Trained to maximize value V(x_t)'''
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
    ''' Critic Network for ADHDP: V: x_t -> V_hat (value/reward-to-go) [float]'''
    def __init__(self, obs_dim: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
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
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer
        output_layer = nn.Linear(prev_size, obs_dim)
        nn.init.uniform_(output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(output_layer.bias, -3e-3, 3e-3)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
class HDPAgent():
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int], actor_lr: float, critic_lr: float, model_lr: float, gamma: float, device: str, action_low: np.ndarray = None, action_high: np.ndarray = None):
        self.device = device
        self.gamma = gamma

        self.actor = Actor(obs_dim, act_dim, hidden_sizes).to(device)
        self.critic = Critic(obs_dim, hidden_sizes).to(device)
        self.model = Model(obs_dim, act_dim, hidden_sizes).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=model_lr)

        # Store action space bounds for scaling
        self.action_low = action_low if action_low is not None else -np.ones(act_dim)
        self.action_high = action_high if action_high is not None else np.ones(act_dim)

        # Memory
        self.prev_obs = None
        self.prev_reward = None
        self.next_obs_pred = None
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():  # Disable gradient tracking during inference
            action = self.actor(obs)
        action = action.cpu().numpy().squeeze(0)  # Remove batch dimension

        # Scale from [-1, 1] to [action_low, action_high]
        action_scaled = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)

        return action_scaled

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, next_obs: np.ndarray) -> dict:
        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Handle first step (no previous observation)
        if self.prev_obs is None:
            self.prev_obs = obs
            self.prev_reward = reward
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'model_loss': 0.0}

        # FORWARD PASSES

        # Critic: TD error using r_{t-1} + γ*V(x_t) - V(x_{t-1})
        with torch.no_grad():
            critic_target = self.prev_reward + self.gamma * self.critic(obs)
        critic_pred = self.critic(self.prev_obs)
        critic_loss = 0.5 * (critic_target - critic_pred).pow(2).mean()

        # Model: predict next state from actual state-action pair that was taken
        # 'action' here is the action from get_action() that led to next_obs
        model_input = torch.cat([obs, action], dim=-1)
        model_pred = self.model(model_input)
        model_target = next_obs
        model_loss = 0.5 * (model_target - model_pred).pow(2).mean()

        # Actor: recompute action from current policy and maximize V(model(x_t, π(x_t)))
        # This creates a new forward pass through actor → model → critic for gradient flow
        actor_action = self.actor(obs)
        actor_model_input = torch.cat([obs, actor_action], dim=-1)
        next_state_pred = self.model(actor_model_input)
        next_value = self.critic(next_state_pred)

        # BACKWARD
        actor_loss = -next_value.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        # Update memory for next iteration
        self.prev_obs = obs.detach()
        self.prev_reward = reward

        # Return losses for logging
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'model_loss': model_loss.item()
        }


if __name__ == "__main__":
    print("Testing HDP Agent Implementation...")

    # Simple dummy environment: inverted pendulum-like
    class DummyEnv:
        def __init__(self):
            self.obs_dim = 4
            self.act_dim = 1
            self.state = None
            self.step_count = 0
            self.max_steps = 200

        def reset(self):
            self.state = np.random.randn(self.obs_dim) * 0.1
            self.step_count = 0
            return self.state

        def step(self, action):
            # Simple dynamics: next_state depends on current state and action
            # Reward encourages state to stay near zero and small actions
            noise = np.random.randn(self.obs_dim) * 0.01
            self.state = 0.95 * self.state + 0.1 * action[0] + noise

            # Reward: negative squared distance from origin and action penalty
            reward = -np.sum(self.state**2) - 0.01 * np.sum(action**2)

            self.step_count += 1
            terminated = self.step_count >= self.max_steps

            return self.state.copy(), reward, terminated

    # Create environment and agent
    env = DummyEnv()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = HDPAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        hidden_sizes=[64, 64],
        actor_lr=1e-3,
        critic_lr=1e-3,
        model_lr=1e-3,
        gamma=0.99,
        device=device
    )

    print(f"Device: {device}")
    print(f"Observation dim: {env.obs_dim}, Action dim: {env.act_dim}")

    # Training loop
    num_episodes = 5
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            # Get action from agent
            action = agent.get_action(obs)

            # Take step in environment
            next_obs, reward, done = env.step(action)

            # Update agent
            agent.update(obs, action, reward, done, next_obs)

            episode_reward += reward
            obs = next_obs
            step += 1

        print(f"Episode {episode + 1}/{num_episodes} - Steps: {step}, Total Reward: {episode_reward:.2f}")

    print("\nHDP Agent test completed successfully!")