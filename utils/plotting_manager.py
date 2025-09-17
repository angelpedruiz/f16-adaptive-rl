"""
Plotting manager for training metrics and test episode trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional


class PlottingManager:
    """
    Handles plotting for:
    1. Training metrics: episode rewards, lengths, and training error.
    2. Test episode trajectories: states, actions, reference overlays, and tracking errors.
    """

    def __init__(self, run_dir: Path, config: Dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.enabled = config.get("enabled", True)
        self.rolling_window = config.get("rolling_window", 50)
        self.dpi = config.get("dpi", 150)
        self.style = config.get("style", "default")
        
        # Create organized plot directories
        self.plots_dir = self.run_dir / "plots"
        self.training_metrics_dir = self.plots_dir / "training_metrics"
        self.trajectories_dir = self.plots_dir / "trajectories"
        
        # Ensure directories exist
        self.training_metrics_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)
        
        if self.style != "default":
            try:
                plt.style.use(self.style)
            except OSError:
                print(f"Warning: Style '{self.style}' not found, using default.")

    def get_training_metrics_path(self, episode: int, is_final: bool = False) -> Path:
        """Get the organized path for training metrics plots."""
        suffix = "_final" if is_final else ""
        return self.training_metrics_dir / f"training_metrics_ep{episode + 1}{suffix}.png"
    
    def get_trajectory_path(self, episode: int = None, name: str = None) -> Path:
        """Get the organized path for trajectory plots."""
        if name:
            filename = f"{name}.png"
        elif episode is not None:
            filename = f"trajectory_ep{episode + 1}.png"
        else:
            filename = "trajectory_test.png"
        return self.trajectories_dir / filename

    @staticmethod
    def _moving_average(arr, window, mode="valid"):
        return np.convolve(np.array(arr).flatten(), np.ones(window), mode=mode) / window

    def create_training_metrics_plot(self, episode: int, env, training_logger, is_final: bool = False) -> None:
        """
        Create training metrics plot with episode rewards, lengths, and training errors
        in a single row of three subplots, using rolling average window.
        Shows all data including from resumed checkpoints for continuity.
        
        Args:
            episode: Current episode number
            env: Environment with RecordEpisodeStatistics wrapper
            training_logger: Training logger with additional statistics
            is_final: Whether this is the final plot
        """
        if not self.enabled:
            return
            
        # Extract data from environment wrapper
        rewards = []
        episode_lengths = []
        
        if hasattr(env, 'return_queue') and len(env.return_queue) > 0:
            rewards = list(env.return_queue)
        
        if hasattr(env, 'length_queue') and len(env.length_queue) > 0:
            episode_lengths = list(env.length_queue)
        
        # Extract training errors from logger or agent
        training_errors = []
        if hasattr(training_logger, 'get_training_stats'):
            stats = training_logger.get_training_stats()
            training_errors = stats.get('training_errors', [])
        elif hasattr(training_logger, 'training_error'):
            training_errors = training_logger.training_error

        fig, axs = plt.subplots(ncols=3, figsize=(12, 5), dpi=self.dpi)
        fig.suptitle(f"Training Metrics (Rolling Window = {self.rolling_window})", fontsize=14)

        # Plot 1: Episode rewards
        if rewards and len(rewards) > 0:
            axs[0].set_title("Episode rewards")
            if len(rewards) >= self.rolling_window:
                reward_moving_average = self._moving_average(rewards, self.rolling_window, "valid")
                axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
            else:
                axs[0].plot(range(len(rewards)), rewards)
        else:
            axs[0].set_title("Episode rewards (No data)")
            axs[0].text(0.5, 0.5, 'No data available', transform=axs[0].transAxes, ha='center', va='center')

        # Plot 2: Episode lengths
        if episode_lengths and len(episode_lengths) > 0:
            axs[1].set_title("Episode lengths")
            if len(episode_lengths) >= self.rolling_window:
                length_moving_average = self._moving_average(episode_lengths, self.rolling_window, "valid")
                axs[1].plot(range(len(length_moving_average)), length_moving_average)
            else:
                axs[1].plot(range(len(episode_lengths)), episode_lengths)
        else:
            axs[1].set_title("Episode lengths (No data)")
            axs[1].text(0.5, 0.5, 'No data available', transform=axs[1].transAxes, ha='center', va='center')

        # Plot 3: Training error
        if training_errors and len(training_errors) > 0:
            axs[2].set_title("Training Error")
            if len(training_errors) >= self.rolling_window:
                training_error_moving_average = self._moving_average(training_errors, self.rolling_window, "same")
                axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
            else:
                axs[2].plot(range(len(training_errors)), training_errors)
        else:
            axs[2].set_title("Training Error (No data)")
            axs[2].text(0.5, 0.5, 'No data available', transform=axs[2].transAxes, ha='center', va='center')

        plt.tight_layout()
        save_path = self.get_training_metrics_path(episode, is_final)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_test_episode_trajectory(self, states: np.ndarray, actions: np.ndarray,
                                     references: Optional[np.ndarray] = None,
                                     errors: Optional[np.ndarray] = None,
                                     rewards: Optional[List[float]] = None,
                                     episode_num: Optional[int] = None,
                                     save_path: Optional[str] = None,
                                     tracked_indices: Optional[List[int]] = None,
                                     reference_config: Optional[dict] = None) -> None:
        if not self.enabled:
            return

        # Align arrays
        if len(states) > len(actions):
            states = states[:-1]
            if references is not None:
                references = references[:-1]
            if errors is not None:
                errors = errors[:-1]

        state_labels = [
            "Altitude [ft]", "Pitch [rad]", "Velocity [ft/s]",
            "AoA [rad]", "Pitch Rate [rad/s]", "Thrust [lbs]", "Elevator [deg]"
        ]
        action_labels = ["Thrust Input [lbs]", "Elevator Input [deg]"]

        t = np.arange(len(actions))

        # Determine which states have references (for overlay plotting)
        states_with_references = []
        if reference_config is not None:
            states_with_references = list(reference_config.keys())
        elif tracked_indices is not None:
            states_with_references = tracked_indices
        elif errors is not None:
            states_with_references = list(np.where(np.any(errors != 0, axis=0))[0])
        elif references is not None:
            # Auto-detect by checking which states have non-None references
            for state_idx in range(min(references.shape[1], 7)):
                if any(ref[state_idx] is not None for ref in references if hasattr(ref, '__getitem__')):
                    states_with_references.append(state_idx)

        # Always plot all 7 states + actions + tracking errors for tracked states only
        num_state_plots = 7  # Always plot all 7 states
        num_error_plots = len(states_with_references) if errors is not None else 0
        total_plots = num_state_plots + actions.shape[1] + num_error_plots

        rows, cols = (1, total_plots) if total_plots <= 5 else (2, int(np.ceil(total_plots / 2)))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows), dpi=self.dpi)
        fig.set_facecolor("black")
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        plot_idx = 0

        # Plot all 7 states (always show all states)
        for i in range(7):  # Always plot exactly 7 states
            ax = axes[plot_idx]
            if i < states.shape[1]:  # Only plot if state exists
                ax.plot(t, states[:, i], color="cyan", linewidth=2, label="State")
                
                # Overlay reference if it exists for this state
                if references is not None and i < references.shape[1] and i in states_with_references:
                    # Extract only non-None reference values for plotting
                    ref_values = []
                    ref_times = []
                    for step_idx, ref_val in enumerate(references[:, i]):
                        if ref_val is not None:
                            ref_values.append(ref_val)
                            ref_times.append(step_idx)
                    
                    if ref_values:  # Only plot if we have non-None reference values
                        ax.plot(ref_times, ref_values, color="magenta", linestyle="--", linewidth=2, label="Reference")
                        ax.legend(loc="best", facecolor="black", edgecolor="white", labelcolor="white")
            else:
                # State doesn't exist - show empty plot with label
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', color='white')
            
            ax.set_xlabel("Time Step", color="white")
            ax.set_ylabel(state_labels[i] if i < len(state_labels) else f"State {i}", color="white")
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
            plot_idx += 1

        # Plot actions
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

        # Plot tracking errors (only for states with references)
        if errors is not None and states_with_references:
            for j, idx in enumerate(states_with_references):
                if j < errors.shape[1]:  # Ensure error data exists
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

        # Hide unused axes
        for i in range(plot_idx, len(axes)):
            axes[i].axis("off")

        # Figure title
        ep_str = f"Episode {episode_num + 1}" if episode_num is not None else "Test Episode"
        title = f"{ep_str} — State, Action & Tracking Error Evolution"
        if rewards is not None:
            title += f" (Total Reward: {sum(rewards):.1f})"
        fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.95)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        if save_path is not None:
            # If save_path is provided, use it; otherwise save to trajectories directory
            if isinstance(save_path, str) or isinstance(save_path, Path):
                final_save_path = Path(save_path)
                # Ensure parent directory exists
                final_save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                final_save_path = self.get_trajectory_path(episode_num)
            plt.savefig(final_save_path, facecolor="black", dpi=self.dpi)
            plt.close()
        else:
            plt.show()
