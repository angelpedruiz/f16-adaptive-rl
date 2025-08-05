import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gym
import time

class TD3:
    def __init__(
        self,
        env: gym.Env,
        actor_lr: float,
        critic_lr: float,
        discount_factor: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_delay: int,
        batch_size: int,
        replay_buffer_size: int = int(1e6),
        hidden_sizes: tuple = (256, 256),
        device: str = "cpu",
    ):
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.discount_factor = discount_factor
        self.total_it = 0

        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))
        self.max_action = float(env.action_space.high[0])

        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Actor network and target
        self.actor = self.build_mlp(self.obs_dim, self.action_dim, hidden_sizes, nn.Tanh)
        self.actor_target = self.build_mlp(self.obs_dim, self.action_dim, hidden_sizes, nn.Tanh)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin critic networks and targets
        self.critic_1 = self.build_mlp(self.obs_dim + self.action_dim, 1, hidden_sizes)
        self.critic_2 = self.build_mlp(self.obs_dim + self.action_dim, 1, hidden_sizes)
        self.critic_target_1 = self.build_mlp(self.obs_dim + self.action_dim, 1, hidden_sizes)
        self.critic_target_2 = self.build_mlp(self.obs_dim + self.action_dim, 1, hidden_sizes)

        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.actor.to(device)
        self.actor_target.to(device)
        self.critic_1.to(device)
        self.critic_2.to(device)
        self.critic_target_1.to(device)
        self.critic_target_2.to(device)

    def build_mlp(self, input_dim, output_dim, hidden_sizes, output_activation=None):
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, output_dim))
        if output_activation:
            layers.append(output_activation())
        return nn.Sequential(*layers)

    def scale_action(self, action_normalized: np.ndarray) -> np.ndarray:
        low = self.env.action_space.low
        high = self.env.action_space.high
        return low + (action_normalized + 1.0) * 0.5 * (high - low)

    def unscale_action(self, action: np.ndarray) -> np.ndarray:
        low = self.env.action_space.low
        high = self.env.action_space.high
        return 2.0 * (action - low) / (high - low) - 1.0

    def get_action(self, obs: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        self.actor.eval()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_normalized = self.actor(obs_tensor).cpu().numpy().flatten()
        self.actor.train()

        noise = noise_scale * np.random.randn(self.action_dim)
        action_noisy = action_normalized + noise
        action_noisy = np.clip(action_noisy, -1.0, 1.0)
        return self.scale_action(action_noisy)

    def get_random_action(self):
        """Get a completely random action for exploration"""
        return self.env.action_space.sample()

    def store_transition(self, obs, action, reward, next_obs, done):
        action_norm = self.unscale_action(action)
        self.replay_buffer.append((
            np.array(obs, dtype=np.float32),
            np.array(action_norm, dtype=np.float32),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            bool(done),
        ))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.total_it += 1

        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

        obs = torch.FloatTensor(np.stack(obs_batch)).to(self.device)
        action = torch.FloatTensor(np.stack(action_batch)).to(self.device)
        reward = torch.FloatTensor(np.array(reward_batch)).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(np.stack(next_obs_batch)).to(self.device)
        done = torch.FloatTensor(np.array(done_batch).astype(np.float32)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)

            target_q1 = self.critic_target_1(torch.cat([next_obs, next_action], dim=1))
            target_q2 = self.critic_target_2(torch.cat([next_obs, next_action], dim=1))
            target_q = reward + (1 - done) * self.discount_factor * torch.min(target_q1, target_q2)

        current_q1 = self.critic_1(torch.cat([obs, action], dim=1))
        current_q2 = self.critic_2(torch.cat([obs, action], dim=1))

        critic_loss_1 = nn.functional.mse_loss(current_q1, target_q)
        critic_loss_2 = nn.functional.mse_loss(current_q2, target_q)

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic_1(torch.cat([obs, self.actor(obs)], dim=1)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic_target_1, self.critic_1, self.tau)
            self.soft_update(self.critic_target_2, self.critic_2, self.tau)

    def soft_update(self, target_net, source_net, tau):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def shaped_reward(obs, action, reward, next_obs, done):
    """Add reward shaping to encourage exploration"""
    position, velocity = next_obs[0], next_obs[1]
    
    # Original reward
    shaped_reward = reward
    
    # Height bonus - reward getting higher up the hill
    height_bonus = position * 10.0  # Scale factor
    shaped_reward += height_bonus
    
    # Velocity bonus - reward having speed in the right direction
    if position < 0:  # Left side of hill
        velocity_bonus = max(0, velocity) * 5.0  # Reward positive velocity
    else:  # Right side of hill
        velocity_bonus = max(0, -velocity) * 5.0  # Reward negative velocity
    shaped_reward += velocity_bonus
    
    # Energy penalty - penalize excessive force to encourage efficiency
    energy_penalty = -0.1 * (action[0] ** 2)
    shaped_reward += energy_penalty
    
    return shaped_reward


# === Training ===

env = gym.make("MountainCarContinuous-v0")

agent = TD3(
    env,
    actor_lr=1e-3,
    critic_lr=1e-3,
    discount_factor=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_delay=2,
    batch_size=64,
    device="cpu",
)

max_steps = 999  # Use the environment's natural max steps
warmup = 2000  # More random exploration at the start
num_episodes = 300  # Train longer
random_episodes = 50  # Pure random exploration episodes

successful_episodes = 0
best_reward = -float('inf')

for episode in range(num_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    original_reward = 0
    start_time = time.time()
    
    # Exploration strategy
    if episode < random_episodes:
        # Pure random exploration for first episodes
        use_random = True
        noise_scale = 0.0
    else:
        use_random = False
        # Higher noise early in training, decay over time
        noise_scale = max(0.3 * (1 - (episode - random_episodes) / (num_episodes - random_episodes)), 0.05)
    
    max_position = -1.2  # Track highest position reached
    
    for step in range(max_steps):
        if use_random:
            action = agent.get_random_action()
        else:
            action = agent.get_action(obs, noise_scale=noise_scale)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Track progress
        max_position = max(max_position, next_obs[0])
        original_reward += reward
        
        # Apply reward shaping
        shaped_reward_val = shaped_reward(obs, action, reward, next_obs, done)
        
        agent.store_transition(obs, action, shaped_reward_val, next_obs, done)
        
        if len(agent.replay_buffer) > warmup:
            agent.update()
        
        obs = next_obs
        episode_reward += shaped_reward_val
        
        if done:
            break
    
    # Track success
    if original_reward > 90:  # Close to the +100 goal reward
        successful_episodes += 1
    
    if original_reward > best_reward:
        best_reward = original_reward
    
    elapsed = time.time() - start_time
    
    if episode % 10 == 0 or episode < 10:
        print(f"Episode {episode + 1}: "
              f"original_reward={original_reward:.1f}, "
              f"shaped_reward={episode_reward:.1f}, "
              f"max_pos={max_position:.3f}, "
              f"noise={noise_scale:.3f}, "
              f"successes={successful_episodes}, "
              f"best={best_reward:.1f}, "
              f"buffer_size={len(agent.replay_buffer)}, "
              f"time={elapsed:.1f}s")

print(f"\nTraining completed! Best reward: {best_reward:.1f}, Successful episodes: {successful_episodes}")

# Test the final policy
print("\nTesting final policy...")
test_episodes = 5
test_rewards = []

for test_ep in range(test_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    max_pos = -1.2
    
    for step in range(max_steps):
        action = agent.get_action(obs, noise_scale=0.0)  # No noise for testing
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        max_pos = max(max_pos, next_obs[0])
        obs = next_obs
        episode_reward += reward
        
        if done:
            break
    
    test_rewards.append(episode_reward)
    print(f"Test episode {test_ep + 1}: reward = {episode_reward:.1f}, max_position = {max_pos:.3f}")

print(f"Average test reward: {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")

# Render one final episode
if np.mean(test_rewards) > 50:  # Only render if doing reasonably well
    print("\nRendering final episode...")
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    obs, _ = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        env.render()
        action = agent.get_action(obs, noise_scale=0.0)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = next_obs
        episode_reward += reward
        if done:
            break
        time.sleep(0.02)  # Slow down for better visualization
    
    print(f"Rendered episode reward: {episode_reward:.1f}")
    env.close()
else:
    print("Policy not performing well enough for rendering.")

env.close()