def plot_test_episode(states, actions, references, errors, rewards, episode_id, save_path, tracked_indices=None):
    """
    Plots state, action, reference, and tracking error trajectories for a test/evaluation episode.

    Args:
        states (ndarray): (T, state_dim)
        actions (ndarray): (T, action_dim)
        references (ndarray): (T, state_dim) – full reference (zero for untracked states)
        errors (ndarray): (T, num_tracked) – error vectors (only for tracked states)
        rewards (list): Reward per step
        episode_id (int|str): Identifier for the episode (e.g., 0, 5, 'test')
        save_path (Path): Where to save the plot
        tracked_indices (list[int] | None): Indices of states being tracked (must match order in `errors`)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Handle shape mismatches (common if states has one more row than actions)
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

    # Automatically determine tracked indices if not provided
    if tracked_indices is None:
        # Find which states have non-zero references
        tracked_indices = []
        for i in range(references.shape[1]):
            if np.any(np.abs(references[:, i]) > 1e-10):  # Non-zero with tolerance
                tracked_indices.append(i)

    # Calculate MSE for each tracked state
    mse_values = []
    for j, state_idx in enumerate(tracked_indices):
        if j < errors.shape[1]:
            mse = np.mean(errors[:, j] ** 2)
            mse_values.append(mse)
        else:
            mse_values.append(0.0)

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
        
        # Check if this state is tracked and has a non-zero reference
        if i in tracked_indices:
            # Make sure we're plotting the correct reference column
            ref_data = references[:, i]
            if np.any(np.abs(ref_data) > 1e-10):  # Only plot if reference is non-zero
                ax.plot(t, ref_data, color="magenta", linestyle=":", linewidth=2, label="Reference")
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

    # === Plot tracking errors with MSE annotations ===
    for j, state_idx in enumerate(tracked_indices):
        if j < errors.shape[1]:  # Make sure we don't exceed error array bounds
            ax = axes[plot_idx]
            ax.plot(t, errors[:, j], color="red", linestyle="--", linewidth=2)
            label = state_labels[state_idx] if state_idx < len(state_labels) else f"State {state_idx}"
            ax.set_ylabel(f"{label} Tracking Error", color="white")
            ax.set_xlabel("Time Step", color="white")
            
            # Add MSE annotation
            mse_text = f"MSE: {mse_values[j]:.4f}"
            ax.text(0.02, 0.95, mse_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="darkred", alpha=0.8),
                   color="white", fontsize=10, verticalalignment='top')
            
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
            plot_idx += 1

    # === Hide unused subplots ===
    for j in range(plot_idx, len(axes)):
        axes[j].axis("off")

    # Include overall MSE summary in title
    overall_mse = np.mean(mse_values) if mse_values else 0.0
    total_reward = sum(rewards)
    fig.suptitle(
        f"Test Episode {episode_id} — State, Action & Tracking Error Evolution\n"
        f"Total Reward: {total_reward:.1f} | Avg MSE: {overall_mse:.4f}",
        color="white", fontsize=14, fontweight="bold", y=0.95
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    fig.savefig(save_path, facecolor="black", dpi=150)
    plt.close(fig)