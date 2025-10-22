from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from pathlib import Path


def plot_best_action(agent, state_space):
    """
    Plot the best action for each discretized state using the agent's Q-table.
    Annotates each cell with the undiscretized best action value.
    """

    x_bins, y_bins = state_space.bins
    x_edges = np.linspace(
        state_space.lower_bounds[0], state_space.upper_bounds[0], x_bins + 1
    )
    y_edges = np.linspace(
        state_space.lower_bounds[1], state_space.upper_bounds[1], y_bins + 1
    )

    # Meshgrid for pcolormesh (edges, not centers)
    X, Y = np.meshgrid(x_edges, y_edges)

    # Z stores the undiscretized best action (shape: [y_bins, x_bins])
    Z = np.zeros((y_bins, x_bins))

    # Get bin centers for annotation placement
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_centers, y_centers)

    for i in range(x_bins):
        for j in range(y_bins):
            discretized_state = (i, j)
            q_vals = agent.q_values[discretized_state]

            # Use np.allclose to handle floating point precision
            if np.allclose(q_vals, q_vals[0]):
                Z[j, i] = np.nan
                continue

            best_action_idx = np.argmax(q_vals)
            best_action = agent.agent_action_space.undiscretize((best_action_idx,))
            # print(f"Undiscretized state: {discretized_state}, best action: {best_action_idx}, Q-value: {q_vals[best_action_idx]}")
            Z[j, i] = best_action[0]

    # Plotting
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    c = plt.pcolormesh(X, Y, Z, cmap=cmap, shading="auto")
    plt.colorbar(c, label="Best Action Value")

    # Annotate each cell with the undiscretized action
    for i in range(x_bins):
        for j in range(y_bins):
            plt.text(
                Xc[j, i],
                Yc[j, i],
                f"{Z[j, i]:.2f}",
                ha="center",
                va="center",
                color="white" if Z[j, i] < (np.max(Z) + np.min(Z)) / 2 else "black",
                fontsize=8,
            )

    plt.xlabel("Theta (Position)")
    plt.ylabel("Theta Dot (Velocity)")
    plt.title("Best Action")
    plt.tight_layout()
    plt.show()


def reward_length_learning_error_plot(env, agent, rolling_length=500, save_path=None):
    def get_moving_avgs(arr, window, convolution_mode):
        return (
            np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode)
            / window
        )

    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    fig.suptitle(f"Training Metrics (Rolling Window = {rolling_length})", fontsize=14)

    # Plot: Rewards
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    # Plot: Episode lengths
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    # Plot: Training error
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error, rolling_length, "same"
    )
    axs[2].plot(
        range(len(training_error_moving_average)), training_error_moving_average
    )

    plt.tight_layout()

    if save_path is not None:
        # Ensure the directory exists if save_path is provided
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def multi_agent_reward_length_learning_error_plot(
    agent_env_pairs: list[tuple], rolling_length: int
):
    def get_moving_avgs(arr, window, convolution_mode):
        return (
            np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode)
            / window
        )

    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    axs[1].set_title("Episode lengths")
    axs[2].set_title("Training Error")

    for env, agent in agent_env_pairs:
        label = f"{agent.__class__.__name__}"

        # Plot rewards
        reward_moving_average = get_moving_avgs(
            env.return_queue, rolling_length, "valid"
        )
        axs[0].plot(
            range(len(reward_moving_average)), reward_moving_average, label=label
        )

        # Plot lengths
        length_moving_average = get_moving_avgs(
            env.length_queue, rolling_length, "valid"
        )
        axs[1].plot(
            range(len(length_moving_average)), length_moving_average, label=label
        )

        # Plot training error
        training_error_moving_average = get_moving_avgs(
            agent.training_error, rolling_length, "same"
        )
        axs[2].plot(
            range(len(training_error_moving_average)),
            training_error_moving_average,
            label=label,
        )

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_test_episode(agent, env, n_steps=200, action_signal=None):
    """
    Plots the state dimensions and actions over time for a test episode of the aircraft.

    Parameters:
    - agent: The trained Q-learning agent.
    - env: The aircraft environment.
    - n_steps: The number of steps to run in the test episode.
    """
    obs, _ = env.reset()  # Reset the environment and get the initial observation
    states = []  # List to store states
    actions = []  # List to store actions
    rewards = []  # List to store rewards

    print(n_steps)
    for step in range(n_steps):
        # Get action from the agent
        discretized_obs = agent.agent_state_space.discretize(obs)

        # Select action (with tie-breaking for equal Q-values)
        if action_signal is not None:
            # If action signal is provided, use it
            action = action_signal(step)
        else:
            # Otherwise, choose the best action based on Q-values
            action = (
                np.random.choice(
                    np.flatnonzero(
                        agent.q_values[discretized_obs]
                        == np.max(agent.q_values[discretized_obs])
                    )
                ),
            )
            action = agent.agent_action_space.undiscretize(action)

        # Take a step in the environment
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Store the state, action, and reward
        states.append(obs)
        actions.append(action)
        rewards.append(reward)

        # Move to the next state
        obs = next_obs

        if terminated or truncated:
            break  # End the episode if it is terminated or truncated

    # Convert states and actions to numpy arrays for plotting
    states = np.array(states)
    actions = np.array(actions)
    t_s = np.arange(len(states))  # Time steps

    # Print dimensions for debugging
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")

    # Define state labels and titles for aircraft model
    # States: x = [Δh, Δθ, Δv, Δα, Δq, Δδ_t, Δδ_e, tracking_error]
    default_state_labels = [
        "Δh [ft]",  # Altitude deviation
        "Δθ [deg]",  # Pitch angle deviation
        "Δv [ft/s]",  # Velocity deviation
        "Δα [deg]",  # Angle of attack deviation
        "Δq [deg/s]",  # Pitch rate deviation
        "Δδ_t [lbs]",  # Thrust deviation
        "Δδ_e [deg]",  # Elevator angle deviation
        "Tracking Error",  # Tracking error
    ]

    default_state_titles = [
        "Altitude Deviation",
        "Pitch Angle Deviation",
        "Velocity Deviation",
        "Angle of Attack Deviation",
        "Pitch Rate Deviation",
        "Thrust Deviation",
        "Elevator Angle Deviation",
        "Tracking Error",
    ]

    # Actions: u = [Δδ_t, Δδ_e]
    default_action_labels = [
        "Δδ_t [lbs]",  # Thrust command
        "Δδ_e [deg]",  # Elevator command
    ]

    default_action_titles = ["Thrust Command", "Elevator Command"]

    # Adjust labels to match actual dimensions
    num_states = states.shape[1] if len(states.shape) > 1 else 1
    num_actions = actions.shape[1] if len(actions.shape) > 1 else 1

    # Create state labels (use defaults or generate generic ones)
    state_labels = []
    state_titles = []
    for i in range(num_states):
        if i < len(default_state_labels):
            state_labels.append(default_state_labels[i])
            state_titles.append(default_state_titles[i])
        else:
            state_labels.append(f"State {i + 1}")
            state_titles.append(f"State {i + 1} Over Time")

    # Create action labels (use defaults or generate generic ones)
    action_labels = []
    action_titles = []
    for i in range(num_actions):
        if i < len(default_action_labels):
            action_labels.append(default_action_labels[i])
            action_titles.append(default_action_titles[i])
        else:
            action_labels.append(f"Action {i + 1}")
            action_titles.append(f"Action {i + 1} Over Time")

    # Prepare plot data - combine states and actions
    plot_data = []

    # Add state data
    if len(states.shape) > 1:
        for i in range(states.shape[1]):
            plot_data.append((states[:, i], state_labels[i], state_titles[i]))
    else:
        # Handle case where states is 1D
        plot_data.append((states, state_labels[0], state_titles[0]))

    # Add action data
    if len(actions.shape) > 1:
        for i in range(actions.shape[1]):
            plot_data.append((actions[:, i], action_labels[i], action_titles[i]))
    else:
        # Handle case where actions is 1D
        plot_data.append((actions, action_labels[0], action_titles[0]))

    # Create a single large figure with all plots
    # For 10 plots (8 states + 2 actions), use a 5x2 layout
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 5 rows, 2 columns
    fig.set_facecolor("black")

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Plot all data
    for i, (data, ylabel, title) in enumerate(plot_data):
        ax = axes_flat[i]
        ax.plot(t_s, data, color="yellow", linewidth=2)
        ax.set_xlabel("Time Steps", color="white")
        ax.set_ylabel(ylabel, color="white")
        # ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("black")
        ax.tick_params(colors="white")

        # Set spines color
        for spine in ax.spines.values():
            spine.set_color("white")

    # Hide unused subplots (if any)
    for i in range(len(plot_data), len(axes_flat)):
        axes_flat[i].axis("off")

    # Add figure title
    fig.suptitle(
        "Aircraft Test Episode - All States and Actions",
        color="white",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the main title
    plt.show()

    # Print episode summary
    print(f"\nEpisode Summary:")
    print(f"Total steps: {len(states)}")
    print(f"Total reward: {sum(rewards):.2f}")
    print(f"Average reward per step: {np.mean(rewards):.3f}")


def plot_episode_lunar_lander(agent, env):
    obs, info = env.reset()
    done = False
    episode_return = 0.0
    while not done:
        env.render()
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
        done = terminated or truncated
        episode_return += reward
    # print('Score:{}'.format(info['episode']['r']))
    env.close()
    print(f"Episode finished with score: {episode_return:.2f}")

def plot_training_milestone_from_data(states, actions, references, errors, rewards, episode_num, save_path, tracked_indices=None):
    """
    Plots state, action, reference, and tracking error trajectories during a training milestone episode.
    
    Args:
        states (ndarray): (T, state_dim)
        actions (ndarray): (T, action_dim)
        references (ndarray): (T, state_dim) – full reference (zero for untracked states)
        errors (ndarray): (T, num_tracked) – error vectors (only for tracked states)
        rewards (list): Reward per step
        episode_num (int): Which episode this is
        save_path (Path): Save location
        tracked_indices (list[int]): Indices of states being tracked (must match order in `errors`)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if len(states) > len(actions):
        states = states[:-1]
        references = references[:-1]
        errors = errors[:-1]

    # === Labels ===
    state_labels = [
        "Altitude [ft]", "Pitch [rad]", "Velocity [ft/s]",
        "AoA [rad]", "Pitch Rate [rad/s]", "Thrust [lbs]", "Elevator [deg]"
    ]
    action_labels = [
        "Thrust Input [lbs]", "Elevator Input [deg]"
    ]

    t = np.arange(len(actions))
    
    if tracked_indices is None:
        # Fallback: try to infer tracked states where error is not all zero
        tracked_indices = np.where(np.any(errors != 0, axis=0))[0].tolist()

    # === Total plots: states + actions + tracking errors ===
    total_plots = states.shape[1] + actions.shape[1] + len(tracked_indices)

    if total_plots <= 5:
        fig, axes = plt.subplots(1, total_plots, figsize=(3 * total_plots, 4))
    else:
        rows = 2
        cols = int(np.ceil(total_plots / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))

    fig.set_facecolor("black")
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    plot_idx = 0

    # === Plot states with reference overlay if tracked ===
    for i in range(states.shape[1]):
        ax = axes[plot_idx]
        ax.plot(t, states[:, i], color="cyan", linewidth=2, label="Actual")
        if tracked_indices and i in tracked_indices:
            ax.plot(t, references[:, i], color="magenta", linestyle=":", linewidth=2, label="Reference")
            ax.legend(loc="best", facecolor="black", edgecolor="white", labelcolor="white")
        ax.set_xlabel("Time Step", color="white")
        ax.set_ylabel(state_labels[i] if i < len(state_labels) else f"State {i}", color="white")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        plot_idx += 1

    # === Plot actions ===
    for i in range(actions.shape[1]):
        ax = axes[plot_idx]
        ax.plot(t, actions[:, i], color="yellow", linewidth=2)
        ax.set_xlabel("Time Step", color="white")
        ax.set_ylabel(action_labels[i] if i < len(action_labels) else f"Action {i}", color="white")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        plot_idx += 1

    # === Plot tracking errors ===
    for j, idx in enumerate(tracked_indices):
        ax = axes[plot_idx]
        ax.plot(t, errors[:, j], color="red", linestyle="--", linewidth=2)
        label = state_labels[idx] if idx < len(state_labels) else f"State {idx}"
        ax.set_ylabel(f"{label} Tracking Error", color="white")
        ax.set_xlabel("Time Step", color="white")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        plot_idx += 1

    # === Hide unused subplots ===
    for j in range(plot_idx, len(axes)):
        axes[j].axis("off")

    total_reward = sum(rewards)
    fig.suptitle(
        f"Episode {episode_num+1} — State, Action & Tracking Error Evolution (Total Reward: {total_reward:.1f})",
        color="white", fontsize=14, fontweight="bold", y=0.95
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    # Ensure the directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, facecolor="black", dpi=150)
    plt.close(fig)
