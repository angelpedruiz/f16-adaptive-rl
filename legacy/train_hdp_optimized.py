"""
Train HDP agent on PendulumCart with optimized hyperparameters.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from envs.pendulumcart import PendulumCartEnv
from agents.hdp import HDPAgent

def train_hdp_optimized(num_episodes=200, max_steps=200):
    """Train HDP agent with optimized hyperparameters."""

    # Create environment
    env = PendulumCartEnv()

    # Initialize HDP agent with optimized hyperparameters
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
        hidden_sizes=[64, 32],  # Smaller second layer
        actor_lr=1e-4,          # Lower actor learning rate for stability
        critic_lr=5e-3,         # Higher critic learning rate
        model_lr=1e-2,          # Highest for supervised learning
        gamma=0.95,             # Lower gamma for shorter horizon
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

    print(f"\nTraining HDP agent with optimized hyperparameters for {num_episodes} episodes...")
    print(f"  Actor LR:  {1e-4:.0e}")
    print(f"  Critic LR: {5e-3:.0e}")
    print(f"  Model LR:  {1e-2:.0e}")
    print(f"  Gamma:     {0.95}")
    print(f"  Network:   [64, 32]")
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
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_length = np.mean(episode_lengths[-20:])
            avg_actor_loss = np.mean([l for ep_losses in [episode_actor_loss] for l in ep_losses[-20:]])
            print(f"Episode {episode + 1:3d}/{num_episodes} | "
                  f"Reward: {avg_reward:6.2f} | "
                  f"Length: {avg_length:5.1f} | "
                  f"Actor Loss: {avg_actor_loss:6.2f}")

    print("=" * 60)
    print("Training complete!")
    print(f"Final avg reward (last 20 eps): {np.mean(episode_rewards[-20:]):.2f}")
    print(f"Final avg length (last 20 eps): {np.mean(episode_lengths[-20:]):.1f}")

    return {
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'model_losses': model_losses,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

def plot_training_results(metrics, filename='hdp_optimized_results.png'):
    """Create 4-subplot visualization of training progress."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HDP Agent Training (Optimized) on Inverted Pendulum Cart',
                 fontsize=16, fontweight='bold')

    # Prepare step indices
    steps = np.arange(len(metrics['actor_losses']))

    # Plot 1: Actor Loss
    ax1 = axes[0, 0]
    ax1.plot(steps, metrics['actor_losses'], alpha=0.4, linewidth=0.5, label='Actor Loss')
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
    ax2.plot(steps, metrics['critic_losses'], alpha=0.4, linewidth=0.5, label='Critic Loss')
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
    ax3.plot(steps, metrics['model_losses'], alpha=0.4, linewidth=0.5, label='Model Loss')
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
    ax4.plot(episodes, metrics['episode_rewards'], alpha=0.4, linewidth=0.5, label='Episode Reward')
    # Add moving average
    reward_window = min(20, len(metrics['episode_rewards']) // 5)
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
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {filename}")

    plt.show()

if __name__ == "__main__":
    # Train agent with optimized hyperparameters
    metrics = train_hdp_optimized(num_episodes=200, max_steps=200)

    # Visualize results
    plot_training_results(metrics)
