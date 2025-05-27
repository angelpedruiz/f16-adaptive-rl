from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from functools import partial


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


def reward_length_learning_error_plot(env, agent, rolling_length):
    def get_moving_avgs(arr, window, convolution_mode):
        return (
            np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode)
            / window
        )

    # Smooth over a 500 episode window
    rolling_length = rolling_length
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error, rolling_length, "same"
    )
    axs[2].plot(
        range(len(training_error_moving_average)), training_error_moving_average
    )
    plt.tight_layout()
    plt.show()
    # Automatically print agent hyperparameters if they exist
    print("\nHyperparameters used:")
    for attr in dir(agent):
        if not attr.startswith("_") and not callable(getattr(agent, attr)):
            value = getattr(agent, attr)
            # Optionally skip long lists or internal buffers
            if isinstance(value, (int, float, str, bool)):
                print(f"  {attr}: {value}")


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
            action = (np.random.choice(np.flatnonzero(agent.q_values[discretized_obs] == np.max(agent.q_values[discretized_obs]))),)
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
        'Δh [ft]',           # Altitude deviation
        'Δθ [deg]',          # Pitch angle deviation
        'Δv [ft/s]',         # Velocity deviation
        'Δα [deg]',          # Angle of attack deviation
        'Δq [deg/s]',        # Pitch rate deviation
        'Δδ_t [lbs]',        # Thrust deviation
        'Δδ_e [deg]',        # Elevator angle deviation
        'Tracking Error'     # Tracking error
    ]
    
    default_state_titles = [
        'Altitude Deviation',
        'Pitch Angle Deviation',
        'Velocity Deviation', 
        'Angle of Attack Deviation',
        'Pitch Rate Deviation',
        'Thrust Deviation',
        'Elevator Angle Deviation',
        'Tracking Error'
    ]
    
    # Actions: u = [Δδ_t, Δδ_e]
    default_action_labels = [
        'Δδ_t [lbs]',        # Thrust command
        'Δδ_e [deg]'         # Elevator command
    ]
    
    default_action_titles = [
        'Thrust Command',
        'Elevator Command'
    ]
    
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
            state_labels.append(f'State {i+1}')
            state_titles.append(f'State {i+1} Over Time')
    
    # Create action labels (use defaults or generate generic ones)
    action_labels = []
    action_titles = []
    for i in range(num_actions):
        if i < len(default_action_labels):
            action_labels.append(default_action_labels[i])
            action_titles.append(default_action_titles[i])
        else:
            action_labels.append(f'Action {i+1}')
            action_titles.append(f'Action {i+1} Over Time')
    
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
    fig.set_facecolor('black')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Plot all data
    for i, (data, ylabel, title) in enumerate(plot_data):
        ax = axes_flat[i]
        ax.plot(t_s, data, color='yellow', linewidth=2)
        ax.set_xlabel('Time Steps', color='white')
        ax.set_ylabel(ylabel, color='white')
        #ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        
        # Set spines color
        for spine in ax.spines.values():
            spine.set_color('white')
    
    # Hide unused subplots (if any)
    for i in range(len(plot_data), len(axes_flat)):
        axes_flat[i].axis('off')
    
    # Add figure title
    fig.suptitle('Aircraft Test Episode - All States and Actions', 
                color='white', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the main title
    plt.show()
    
    # Print episode summary
    print(f"\nEpisode Summary:")
    print(f"Total steps: {len(states)}")
    print(f"Total reward: {sum(rewards):.2f}")
    print(f"Average reward per step: {np.mean(rewards):.3f}")


