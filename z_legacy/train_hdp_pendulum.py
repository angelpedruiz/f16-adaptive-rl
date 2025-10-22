"""
Train HDP agent on PendulumCart environment and visualize learning progress.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from envs.pendulumcart import PendulumCartEnv
from agents.hdp import HDPAgent

def train_hdp(num_episodes=100, max_steps=200):
    """Train HDP agent and track losses and rewards."""

    # Create environment
    env = PendulumCartEnv()

    # Initialize HDP agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set deterministic seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    agent = HDPAgent(
        obs_dim=env.state_dim,
        act_dim=env.act_dim,
        hidden_sizes=[32, 32],
        actor_lr=5e-1,
        critic_lr=5e-1,
        model_lr=5e-1,
        gamma=0.8,
        device=device,
        action_low=np.array([-env.max_force]),
        action_high=np.array([env.max_force])
    )

    # Tracking metrics
    actor_losses = []
    critic_losses = []
    model_losses = []
    episode_rewards = []
    episode_lengths = []

    print(f"\nTraining HDP agent for {num_episodes} episodes...")
    print("=" * 60)

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_actor_loss = []
        episode_critic_loss = []
        episode_model_loss = []

        for step in range(max_steps):
            # Get action
            action = agent.get_action(obs)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Update agent
            losses = agent.update(obs, action, reward, terminated, next_obs)

            # Track losses
            episode_actor_loss.append(losses['actor_loss'])
            episode_critic_loss.append(losses['critic_loss'])
            episode_model_loss.append(losses['model_loss'])

            episode_reward += reward
            obs = next_obs

            if terminated or truncated:
                break

        # Store episode metrics
        actor_losses.extend(episode_actor_loss)
        critic_losses.extend(episode_critic_loss)
        model_losses.extend(episode_model_loss)
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f}")

    print("=" * 60)
    print("Training complete!")

    return {
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'model_losses': model_losses,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

def plot_training_results(metrics):
    """Create 4-subplot visualization of training progress."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HDP Agent Training on Inverted Pendulum Cart', fontsize=16, fontweight='bold')

    # Prepare step indices
    steps = np.arange(len(metrics['actor_losses']))

    # Plot 1: Actor Loss
    ax1 = axes[0, 0]
    ax1.plot(steps, metrics['actor_losses'], alpha=0.6, linewidth=0.8, label='Actor Loss')
    # Add moving average
    window = min(100, len(metrics['actor_losses']) // 10)
    if window > 0:
        actor_ma = np.convolve(metrics['actor_losses'], np.ones(window)/window, mode='valid')
        ax1.plot(steps[window-1:], actor_ma, color='red', linewidth=2, label=f'MA({window})')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Actor Loss vs Steps')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Critic Loss
    ax2 = axes[0, 1]
    ax2.plot(steps, metrics['critic_losses'], alpha=0.6, linewidth=0.8, label='Critic Loss')
    # Add moving average
    if window > 0:
        critic_ma = np.convolve(metrics['critic_losses'], np.ones(window)/window, mode='valid')
        ax2.plot(steps[window-1:], critic_ma, color='red', linewidth=2, label=f'MA({window})')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title('Critic Loss vs Steps')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Model Loss
    ax3 = axes[1, 0]
    ax3.plot(steps, metrics['model_losses'], alpha=0.6, linewidth=0.8, label='Model Loss')
    # Add moving average
    if window > 0:
        model_ma = np.convolve(metrics['model_losses'], np.ones(window)/window, mode='valid')
        ax3.plot(steps[window-1:], model_ma, color='red', linewidth=2, label=f'MA({window})')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Loss')
    ax3.set_title('Model Loss vs Steps')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Episode Rewards
    ax4 = axes[1, 1]
    episodes = np.arange(1, len(metrics['episode_rewards']) + 1)
    ax4.plot(episodes, metrics['episode_rewards'], alpha=0.6, linewidth=0.8, label='Episode Reward')
    # Add moving average
    reward_window = min(10, len(metrics['episode_rewards']) // 5)
    if reward_window > 0:
        reward_ma = np.convolve(metrics['episode_rewards'], np.ones(reward_window)/reward_window, mode='valid')
        ax4.plot(episodes[reward_window-1:], reward_ma, color='red', linewidth=2, label=f'MA({reward_window})')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Total Reward')
    ax4.set_title('Episode Reward vs Episodes')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    # Save figure
    save_path = 'hdp_training_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    plt.show()

if __name__ == "__main__":
    # Train agent
    metrics = train_hdp(num_episodes=1, max_steps=200)

    # Visualize results
    plot_training_results(metrics)
