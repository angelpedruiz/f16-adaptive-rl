from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

def plot_best_action(agent, state_space):
    """
    Plot the best action for each discretized state using the agent's Q-table.
    Annotates each cell with the undiscretized best action value.
    """

    x_bins, y_bins = state_space.bins
    x_edges = np.linspace(state_space.lower_bounds[0], state_space.upper_bounds[0], x_bins + 1)
    y_edges = np.linspace(state_space.lower_bounds[1], state_space.upper_bounds[1], y_bins + 1)

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
            #print(f"Undiscretized state: {discretized_state}, best action: {best_action_idx}, Q-value: {q_vals[best_action_idx]}")
            Z[j, i] = best_action[0]


    # Plotting
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('viridis')
    c = plt.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')
    plt.colorbar(c, label='Best Action Value')

    # Annotate each cell with the undiscretized action
    for i in range(x_bins):
        for j in range(y_bins):
            plt.text(
                Xc[j, i], Yc[j, i],
                f'{Z[j, i]:.2f}',
                ha='center', va='center',
                color='white' if Z[j, i] < (np.max(Z) + np.min(Z)) / 2 else 'black',
                fontsize=8
            )

    plt.xlabel('Theta (Position)')
    plt.ylabel('Theta Dot (Velocity)')
    plt.title('Best Action')
    plt.tight_layout()
    plt.show()

def reward_length_learning_error_plot(env, agent, rolling_length):
    def get_moving_avgs(arr, window, convolution_mode):
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    # Smooth over a 500 episode window
    rolling_length = rolling_length
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
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

def multi_agent_reward_length_learning_error_plot(agent_env_pairs: list[tuple], rolling_length: int):
    def get_moving_avgs(arr, window, convolution_mode):
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    axs[1].set_title("Episode lengths")
    axs[2].set_title("Training Error")

    for env, agent in agent_env_pairs:
        label = f'{agent.__class__.__name__}'
        
        # Plot rewards
        reward_moving_average = get_moving_avgs(
            env.return_queue,
            rolling_length,
            "valid"
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average, label=label)

        # Plot lengths
        length_moving_average = get_moving_avgs(
            env.length_queue,
            rolling_length,
            "valid"
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average, label=label)

        # Plot training error
        training_error_moving_average = get_moving_avgs(
            agent.training_error,
            rolling_length,
            "same"
        )
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average, label=label)

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_test_episode(agent, env, n_steps=200):
    """
    Plots the state dimensions (theta, theta_dot) and actions over time for a test episode.
    
    Parameters:
    - agent: The trained Q-learning agent.
    - env: The environment (InvertedPendulumEnv).
    - n_steps: The number of steps to run in the test episode.
    """
    obs, _ = env.reset()  # Reset the environment and get the initial observation
    states = []  # List to store states
    actions = []  # List to store actions
    rewards = []  # List to store rewards
    
    for step in range(n_steps):
        # Get action from the agent
        discretized_obs = agent.agent_state_space.discretize(obs)

        #action = (np.argmax(agent.q_values[discretized_obs]),)
        action = (np.random.choice(np.flatnonzero(agent.q_values[discretized_obs] == np.max(agent.q_values[discretized_obs]))),)


        action = agent.agent_action_space.undiscretize(action)  # <-- note the comma!
        
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

    # Plot the state dimensions (theta and theta_dot) over time
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Plot theta (position) over time
    axs[0].plot(range(len(states)), states[:, 0], label="Theta (Position)")
    axs[0].set_title("Theta (Position) Over Time")
    axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Theta (Position)")
    axs[0].legend()

    # Plot theta_dot (velocity) over time
    axs[1].plot(range(len(states)), states[:, 1], label="Theta Dot (Velocity)")
    axs[1].set_title("Theta Dot (Velocity) Over Time")
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("Theta Dot (Velocity)")
    axs[1].legend()

    # Plot actions over time
    axs[2].plot(range(len(actions)), actions, label="Action", color='orange')
    axs[2].set_title("Action Over Time")
    axs[2].set_xlabel("Time Steps")
    axs[2].set_ylabel("Action")
    axs[2].legend()

    plt.tight_layout()
    plt.show()