"""
Test script to verify IHDP linear feedback controller and critic learning.

This script tests:
1. Linear feedback controller can stabilize the system
2. Critic learns reasonable value function with linear feedback actions
"""
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from envs.pendulumcart.pendulumcart import PendulumCartEnv
from agents.IHDP.ihdp import IHDPAgent


def test_linear_feedback_control(env, agent, max_steps=500):
    """
    Test if the linear feedback controller can stabilize the pendulum.

    Returns:
        states, actions, rewards: Arrays of trajectory data
    """
    print("\n" + "="*60)
    print("TEST 1: Linear Feedback Controller Stabilization")
    print("="*60)

    obs, _ = env.reset()
    states = [env.state.copy()]
    actions = []
    rewards = []

    for step in range(max_steps):
        # Get action using linear feedback
        action = agent.get_action(obs, is_linear_feedback=True)

        # Take step in environment
        next_obs, reward, terminated, truncated, _ = env.step(action)

        states.append(env.state.copy())
        actions.append(action)
        rewards.append(reward)

        if terminated or truncated:
            print(f"Episode terminated at step {step}")
            break

        obs = next_obs

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    # Print statistics
    print(f"\nEpisode completed: {len(states)} steps")
    print(f"Final state: x={states[-1,0]:.4f}, theta={states[-1,2]:.4f} rad ({np.degrees(states[-1,2]):.2f}°)")
    print(f"Average reward: {np.mean(rewards):.4f}")
    print(f"Total reward: {np.sum(rewards):.4f}")
    print(f"Max |theta|: {np.max(np.abs(states[:,2])):.4f} rad ({np.degrees(np.max(np.abs(states[:,2]))):.2f}°)")

    # Check if controller stabilized the system
    final_angle = np.abs(states[-1, 2])
    if final_angle < 0.05:  # Within 3 degrees
        print("[PASS] Controller successfully stabilized the pendulum!")
    else:
        print("[FAIL] Controller failed to stabilize (final angle too large)")

    return states, actions, rewards


def test_critic_learning(env, agent, max_steps=1500):
    """
    Test if the critic learns a reasonable value function when using linear feedback.
    Runs multiple episodes with truncation.

    Returns:
        training_data: Dictionary with critic learning metrics
    """
    print("\n" + "="*60)
    print("TEST 2: Critic Learning with Linear Feedback (Multiple Episodes)")
    print("="*60)

    # Initialize storage
    training_data = {
        'critic_predictions': [],
        'critic_targets': [],
        'critic_errors': [],
        'critic_losses': [],
        'rewards': [],
        'states': [],
        'episode_ends': [],  # Track episode boundaries
        'time': [],  # Track actual time
    }

    obs, _ = env.reset()
    agent.prev_obs = torch.FloatTensor(obs).unsqueeze(0)
    agent.prev_reward = torch.FloatTensor([0.0]).unsqueeze(0)

    episode_steps = 0
    total_episodes = 0

    for step in range(max_steps):
        # Get action using linear feedback (to provide consistent control)
        action = agent.get_action(obs, is_linear_feedback=True)

        # Take step in environment
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Update agent (this trains the critic and RLS model)
        metrics = agent.update(obs, action, reward, terminated, next_obs)

        # Store metrics
        training_data['critic_predictions'].append(metrics['critic_prediction'][0])
        training_data['critic_targets'].append(metrics['critic_target'][0])
        training_data['critic_errors'].append(metrics['critic_error'])
        training_data['critic_losses'].append(metrics['losses']['critic_loss'])
        training_data['rewards'].append(reward)
        training_data['states'].append(env.state.copy())
        training_data['time'].append(step * env.dt)  # Actual time in seconds

        episode_steps += 1

        # Print progress every 500 steps
        if (step + 1) % 500 == 0:
            recent_pred = np.mean(training_data['critic_predictions'][-100:])
            recent_target = np.mean(training_data['critic_targets'][-100:])
            recent_error = np.mean(training_data['critic_errors'][-100:])
            recent_reward = np.mean(training_data['rewards'][-100:])
            print(f"Step {step+1:5d} | Episodes: {total_episodes:3d} | V_pred: {recent_pred:7.2f} | V_target: {recent_target:7.2f} | TD_err: {recent_error:.4f} | Reward: {recent_reward:.4f}")

        # Handle episode end
        if terminated or truncated:
            training_data['episode_ends'].append(step)
            total_episodes += 1
            obs, _ = env.reset()
            agent.prev_obs = torch.FloatTensor(obs).unsqueeze(0)
            agent.prev_reward = torch.FloatTensor([0.0]).unsqueeze(0)
            episode_steps = 0
        else:
            obs = next_obs

    # Convert to arrays
    for key in training_data:
        if key != 'episode_ends':
            training_data[key] = np.array(training_data[key])

    # Print final statistics
    print("\n" + "-"*60)
    print("Critic Learning Statistics:")
    print("-"*60)
    print(f"Total episodes completed: {total_episodes}")
    print(f"Total steps: {max_steps}")
    print(f"Average episode length: {max_steps / max(total_episodes, 1):.1f} steps")

    # Check learning in last 500 steps
    last_n = 500
    final_predictions = training_data['critic_predictions'][-last_n:]
    final_targets = training_data['critic_targets'][-last_n:]
    final_errors = training_data['critic_errors'][-last_n:]

    print(f"\nLast {last_n} steps:")
    print(f"  Mean V_prediction: {np.mean(final_predictions):.2f} ± {np.std(final_predictions):.2f}")
    print(f"  Mean V_target:     {np.mean(final_targets):.2f} ± {np.std(final_targets):.2f}")
    print(f"  Mean TD_error:     {np.mean(final_errors):.4f} ± {np.std(final_errors):.4f}")
    print(f"  Mean reward:       {np.mean(training_data['rewards'][-last_n:]):.4f}")

    # Check if critic learned something reasonable
    mean_error = np.mean(final_errors)
    std_error = np.std(final_errors)

    if mean_error < 0.5 and std_error < 1.0:
        print("\n[PASS] Critic appears to have learned a reasonable value function!")
        print(f"  - Low TD error: {mean_error:.4f}")
        print(f"  - Stable predictions (std: {std_error:.4f})")
    else:
        print("\n[FAIL] Critic learning may be unstable or inaccurate")
        print(f"  - TD error: {mean_error:.4f} (target: < 0.5)")
        print(f"  - Prediction std: {std_error:.4f} (target: < 1.0)")

    return training_data


def plot_results(control_states, control_actions, control_rewards,
                critic_data, save_path='test_ihdp_results.png'):
    """
    Plot the test results - all episodes shown continuously as a function of steps.
    """
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)
    fig.suptitle('IHDP Linear Feedback Test Results - All Episodes', fontsize=16, fontweight='bold')

    steps = np.arange(len(critic_data['critic_predictions']))

    # Plot 1: VALUE FUNCTION vs STEPS (MAIN PLOT - spans both columns)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(steps, critic_data['critic_predictions'], label='V(x_t) - Predicted Value',
            color='blue', alpha=0.8, linewidth=1.5)

    # Mark episode boundaries
    for ep_end in critic_data['episode_ends']:
        ax.axvline(x=ep_end, color='red', linestyle='--', alpha=0.3, linewidth=0.8)

    # Add text for episode markers
    if len(critic_data['episode_ends']) > 0:
        ax.text(0.02, 0.98, f"Red dashed lines = Episode boundaries ({len(critic_data['episode_ends'])} episodes)",
                transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Step')
    ax.set_ylabel('Value Function V(x_t)')
    ax.set_title('Predicted Value Function Across All Episodes', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot 2: States across all episodes
    ax = fig.add_subplot(gs[1, :])
    states = critic_data['states']
    ax.plot(steps, states[:, 0], label='x (cart position)', alpha=0.7, linewidth=1)
    ax.plot(steps, states[:, 2], label='θ (angle)', alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Mark episode boundaries
    for ep_end in critic_data['episode_ends']:
        ax.axvline(x=ep_end, color='red', linestyle='--', alpha=0.2, linewidth=0.8)

    ax.set_xlabel('Step')
    ax.set_ylabel('State')
    ax.set_title('States Across All Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Rewards across all episodes
    ax = fig.add_subplot(gs[2, :])
    ax.plot(steps, critic_data['rewards'], label='Reward', alpha=0.7, linewidth=1, color='green')

    # Mark episode boundaries
    for ep_end in critic_data['episode_ends']:
        ax.axvline(x=ep_end, color='red', linestyle='--', alpha=0.2, linewidth=0.8)

    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Rewards Across All Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Critic predictions vs targets
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(steps, critic_data['critic_predictions'], label='V prediction', alpha=0.7, linewidth=1)
    ax.plot(steps, critic_data['critic_targets'], label='V target', alpha=0.7, linewidth=1)

    # Mark episode boundaries
    for ep_end in critic_data['episode_ends']:
        ax.axvline(x=ep_end, color='red', linestyle='--', alpha=0.15, linewidth=0.6)

    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title('Critic Learning: Predictions vs Targets')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: TD error over steps
    ax = fig.add_subplot(gs[3, 1])
    ax.plot(steps, critic_data['critic_errors'], alpha=0.5, linewidth=0.8, color='gray')

    # Add moving average
    window = 50
    if len(critic_data['critic_errors']) > window:
        moving_avg = np.convolve(critic_data['critic_errors'],
                                np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], moving_avg, 'r-', linewidth=2,
               label=f'{window}-step MA', alpha=0.8)

    # Mark episode boundaries
    for ep_end in critic_data['episode_ends']:
        ax.axvline(x=ep_end, color='red', linestyle='--', alpha=0.15, linewidth=0.6)

    ax.set_xlabel('Step')
    ax.set_ylabel('TD Error')
    ax.set_title('Critic Learning: TD Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Critic loss
    ax = fig.add_subplot(gs[4, 0])
    ax.plot(steps, critic_data['critic_losses'], alpha=0.5, linewidth=0.8, color='gray')

    # Add moving average
    if len(critic_data['critic_losses']) > window:
        moving_avg = np.convolve(critic_data['critic_losses'],
                                np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], moving_avg, 'r-', linewidth=2,
               label=f'{window}-step MA', alpha=0.8)

    # Mark episode boundaries
    for ep_end in critic_data['episode_ends']:
        ax.axvline(x=ep_end, color='red', linestyle='--', alpha=0.15, linewidth=0.6)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Critic Learning: Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 7: Histogram of TD errors (last 500 steps)
    ax = fig.add_subplot(gs[4, 1])
    last_n = min(500, len(critic_data['critic_errors']))
    last_errors = critic_data['critic_errors'][-last_n:]
    ax.hist(last_errors, bins=40, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(x=np.mean(last_errors), color='r', linestyle='--',
              linewidth=2, label=f'Mean: {np.mean(last_errors):.4f}')
    ax.set_xlabel('TD Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'TD Error Distribution (last {last_n} steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Results plot saved to: {save_path}")

    return fig


def plot_value_function_detail(critic_data, save_path='value_function_vs_time.png'):
    """
    Create a detailed standalone plot of value function vs time for easy zooming.
    Also includes reward overlay to understand value-reward relationship.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    steps = np.arange(len(critic_data['critic_predictions']))
    time_steps = critic_data['time']

    # Plot 1: Value function with reward overlay
    ax1_twin = ax1.twinx()

    # Main value function plot
    line1 = ax1.plot(time_steps, critic_data['critic_predictions'],
            label='V(x_t) - Predicted Value',
            color='blue', alpha=0.85, linewidth=2)

    # Overlay rewards
    line2 = ax1_twin.plot(time_steps, critic_data['rewards'],
            label='Reward r(x_t)',
            color='green', alpha=0.6, linewidth=1.5)

    # Mark episode boundaries
    for ep_end in critic_data['episode_ends']:
        ep_time = ep_end * 0.01
        ax1.axvline(x=ep_time, color='red', linestyle='--', alpha=0.4, linewidth=1.5,
                   label='Episode boundary' if ep_end == critic_data['episode_ends'][0] else '')

    # Add zero line
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    ax1.set_ylabel('Value Function V(x_t)', fontsize=11, fontweight='bold', color='blue')
    ax1_twin.set_ylabel('Reward r(x_t)', fontsize=11, fontweight='bold', color='green')
    ax1.set_title('Value Function and Rewards Over Time', fontsize=13, fontweight='bold')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='green')

    # Plot 2: Theta (angle) to understand state
    ax2.plot(time_steps, critic_data['states'][:, 2],
            label='θ (pendulum angle)',
            color='orange', alpha=0.8, linewidth=2)

    # Mark episode boundaries
    for ep_end in critic_data['episode_ends']:
        ep_time = ep_end * 0.01
        ax2.axvline(x=ep_time, color='red', linestyle='--', alpha=0.4, linewidth=1.5)

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Angle θ (rad)', fontsize=11, fontweight='bold')
    ax2.set_title('Pendulum Angle (determines reward = -θ²)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Add info box with statistics
    num_episodes = len(critic_data['episode_ends'])
    max_positive_v = np.max(critic_data['critic_predictions'])
    min_reward = np.min(critic_data['rewards'])
    max_reward = np.max(critic_data['rewards'])

    info_text = f"Episodes: {num_episodes}\n"
    info_text += f"Total time: {time_steps[-1]:.1f}s\n"
    info_text += f"Steps: {len(steps)}\n"
    info_text += f"Reward: r = -θ²\n"
    info_text += f"Reward range: [{min_reward:.4f}, {max_reward:.4f}]\n"
    info_text += f"Max V(x): {max_positive_v:.2f} (ERROR: should be ≤ 0!)"

    ax1.text(0.02, 0.98, info_text,
            transform=ax1.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[PLOT] Value function detail plot saved to: {save_path}")
    print(f"\n[WARNING] Critic predicting positive values (max: {max_positive_v:.2f}) but all rewards <= 0!")
    print("[WARNING] This indicates the critic is making systematic errors during learning.")

    return fig


if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("\n" + "="*60)
    print("IHDP Linear Feedback and Critic Learning Test")
    print("="*60)

    # Initialize environment
    dt = 0.01
    max_steps_per_episode = 500
    env = PendulumCartEnv(dt=dt, max_steps=max_steps_per_episode)

    # Initialize IHDP agent
    gamma = 0.99
    agent = IHDPAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        gamma=gamma,
        forgetting_factor=0.99,
        initial_covariance=1.0,
        hidden_sizes={'actor': [6, 6], 'critic': [6, 6]},
        learning_rates={'actor': 5e-2, 'critic': 1e-3},  # Reduced critic LR from 1e-1 to 1e-3
        tau=0.005  # Soft update parameter for target network
    )

    # Test 1: Linear feedback control
    control_states, control_actions, control_rewards = test_linear_feedback_control(
        env, agent, max_steps=500
    )

    # Reset environment and agent for second test
    env = PendulumCartEnv(dt=dt, max_steps=max_steps_per_episode)
    agent = IHDPAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        gamma=gamma,
        forgetting_factor=0.99,
        initial_covariance=1.0,
        hidden_sizes={'actor': [6, 6], 'critic': [6, 6]},
        learning_rates={'actor': 5e-2, 'critic': 1e-3},
        tau=0.005
    )

    # Test 2: Critic learning (3 episodes = ~1500 steps)
    critic_data = test_critic_learning(env, agent, max_steps=1500)

    # Plot results
    plot_results(control_states, control_actions, control_rewards,
                critic_data, save_path='test_ihdp_results.png')

    # Create separate detailed value function plot for easy zooming
    plot_value_function_detail(critic_data, save_path='value_function_vs_time.png')

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
