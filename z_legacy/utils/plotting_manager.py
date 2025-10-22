"""
Plotting manager for training metrics and test episode trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
from gymnasium.utils.save_video import save_video


class PlottingManager:
    """
    Handles plotting for training metrics and environment-specific trajectory plots.

    Features:
    1. Training metrics: episode rewards, lengths, and training error.
    2. Environment-specific trajectory plots:
       - F16: Full state/action/error tracking with reference overlays
       - LunarLander: Position, velocity, and action visualization
       - ShortPeriod: Alpha tracking with reference, pitch rate, and rewards

    Configuration-driven behavior with organized save paths and styling options.
    """

    def __init__(self, run_dir: Path, config: Dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.enabled = config.get("enabled", True)
        self.rolling_window = config['training_metrics'].get("rolling_window", 100)
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

    def _resolve_save_path(self, save_path: Optional[str], episode_num: Optional[int]) -> Optional[Path]:
        """
        Helper method to resolve the save path for trajectory plots.

        Args:
            save_path: User-provided save path or None
            episode_num: Episode number for default naming

        Returns:
            Resolved Path object or None if no saving requested
        """
        if save_path is not None:
            # If save_path is provided, use it
            if isinstance(save_path, str) or isinstance(save_path, Path):
                final_save_path = Path(save_path)
                # Ensure parent directory exists
                final_save_path.parent.mkdir(parents=True, exist_ok=True)
                return final_save_path
            else:
                # If save_path is not a string/Path, use default trajectory path
                return self.get_trajectory_path(episode_num)
        return None

    def _apply_dark_theme(self, ax):
        """Apply consistent dark theme styling to an axis."""
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color("white")

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

    # =========================================================================
    # Environment-Specific Trajectory Plotting Methods
    # =========================================================================

    def plot_f16_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        references: Optional[np.ndarray] = None,
        errors: Optional[np.ndarray] = None,
        rewards: Optional[List[float]] = None,
        episode_num: Optional[int] = None,
        save_path: Optional[str] = None,
        tracked_indices: Optional[List[int]] = None,
        reference_config: Optional[dict] = None
    ) -> None:
        """
        Plot trajectory for F16 environment with 7 states and 2 actions.

        States: Altitude, Pitch, Velocity, AoA, Pitch Rate, Thrust, Elevator
        Actions: Thrust Input, Elevator Input
        """
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
                if any(ref[state_idx] is not None and not np.isnan(ref[state_idx]) for ref in references if hasattr(ref, '__getitem__')):
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
                        if ref_val is not None and not np.isnan(ref_val):
                            ref_values.append(ref_val)
                            ref_times.append(step_idx)

                    if ref_values:  # Only plot if we have non-None reference values
                        ax.plot(ref_times, ref_values, color="magenta", linestyle="--", linewidth=2, label="Reference")
                        ax.legend(loc="best", facecolor="black", edgecolor="white", labelcolor="white")
            else:
                # State doesn't exist - show empty plot with label
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', color='white')

            ax.set_xlabel("Time Step")
            ax.set_ylabel(state_labels[i] if i < len(state_labels) else f"State {i}")
            self._apply_dark_theme(ax)
            plot_idx += 1

        # Plot actions
        for i in range(actions.shape[1]):
            ax = axes[plot_idx]
            ax.plot(t, actions[:, i], color="yellow", linewidth=2)
            ax.set_xlabel("Time Step")
            ax.set_ylabel(action_labels[i] if i < len(action_labels) else f"Action {i}")
            self._apply_dark_theme(ax)
            plot_idx += 1

        # Plot tracking errors (only for states with references)
        if errors is not None and states_with_references:
            errors_array = np.atleast_2d(errors)
            # If errors was 1D, reshape to (N, 1) for consistent indexing
            if errors.ndim == 1:
                errors_array = errors[:, np.newaxis]

            for j, idx in enumerate(states_with_references):
                if j < errors_array.shape[1]:  # Ensure error data exists
                    ax = axes[plot_idx]
                    ax.plot(t, errors_array[:, j], color="red", linestyle="--", linewidth=2)
                    label = state_labels[idx] if idx < len(state_labels) else f"State {idx}"
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel(f"{label} Tracking Error")
                    self._apply_dark_theme(ax)
                    plot_idx += 1

        # Hide unused axes
        for i in range(plot_idx, len(axes)):
            axes[i].axis("off")

        # Figure title
        ep_str = f"Episode {episode_num + 1}" if episode_num is not None else "Test Episode"
        title = f"{ep_str} — F16 State, Action & Tracking Error Evolution"
        if rewards is not None:
            title += f" (Total Reward: {sum(rewards):.1f})"
        fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.95)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        # Save or show plot
        final_save_path = self._resolve_save_path(save_path, episode_num)
        if final_save_path:
            plt.savefig(final_save_path, facecolor="black", dpi=self.dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_lunarlander_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: Optional[List[float]] = None,
        episode_num: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot trajectory for LunarLander environment.

        States: x, y, x_vel, y_vel, angle, angular_vel, left_leg_contact, right_leg_contact
        Actions: main_engine, left/right_orientation
        """
        if not self.enabled:
            return

        # Align arrays
        if len(states) > len(actions):
            states = states[:-1]

        state_labels = [
            "X Position", "Y Position", "X Velocity", "Y Velocity",
            "Angle [rad]", "Angular Velocity", "Left Leg Contact", "Right Leg Contact"
        ]
        action_labels = ["Main Engine", "Left/Right Orientation"]

        t = np.arange(len(actions))

        # Create figure with 2 rows
        fig, axes = plt.subplots(2, 5, figsize=(18, 8), dpi=self.dpi)
        fig.set_facecolor("black")
        axes = axes.flatten()

        plot_idx = 0

        # Plot states
        for i in range(min(8, states.shape[1])):
            ax = axes[plot_idx]
            ax.plot(t, states[:, i], color="cyan", linewidth=2)
            ax.set_xlabel("Time Step")
            ax.set_ylabel(state_labels[i] if i < len(state_labels) else f"State {i}")
            self._apply_dark_theme(ax)
            plot_idx += 1

        # Plot actions
        for i in range(actions.shape[1]):
            ax = axes[plot_idx]
            ax.plot(t, actions[:, i], color="yellow", linewidth=2)
            ax.set_xlabel("Time Step")
            ax.set_ylabel(action_labels[i] if i < len(action_labels) else f"Action {i}")
            self._apply_dark_theme(ax)
            plot_idx += 1

        # Hide unused axes
        for i in range(plot_idx, len(axes)):
            axes[i].axis("off")

        # Figure title
        ep_str = f"Episode {episode_num + 1}" if episode_num is not None else "Test Episode"
        title = f"{ep_str} — LunarLander Trajectory"
        if rewards is not None:
            title += f" (Total Reward: {sum(rewards):.1f})"
        fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.95)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        # Save or show plot
        final_save_path = self._resolve_save_path(save_path, episode_num)
        if final_save_path:
            plt.savefig(final_save_path, facecolor="black", dpi=self.dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # =========================================================================
    # Backward Compatibility
    # =========================================================================

    def plot_test_episode_trajectory(self, states: np.ndarray, actions: np.ndarray,
                                     references: Optional[np.ndarray] = None,
                                     errors: Optional[np.ndarray] = None,
                                     rewards: Optional[List[float]] = None,
                                     episode_num: Optional[int] = None,
                                     save_path: Optional[str] = None,
                                     tracked_indices: Optional[List[int]] = None,
                                     reference_config: Optional[dict] = None) -> None:
        """
        Legacy method for backward compatibility. Defaults to F16 trajectory plotting.

        Deprecated: Use plot_f16_trajectory, plot_lunarlander_trajectory, or
        plot_shortperiod_trajectory instead for clearer environment-specific plotting.
        """
        return self.plot_f16_trajectory(
            states=states,
            actions=actions,
            references=references,
            errors=errors,
            rewards=rewards,
            episode_num=episode_num,
            save_path=save_path,
            tracked_indices=tracked_indices,
            reference_config=reference_config
        )

    # =========================================================================
    # Video Recording
    # =========================================================================

    def save_episode_video(self, env, episode_num: Optional[int] = None) -> None:
        """
        Save a video of the episode from recorded frames.
        
        Args:
            video_frames: List of frames (numpy arrays) recorded during the episode
            episode_num: Optional episode number for naming
            fps: Frames per second for the video
        """
        if not self.enabled:
            return
        
        video_frames = env.render()
        fps = env.metadata.get('render_fps', 30)
        video_filename = f"episode_{episode_num + 1}.mp4" if episode_num is not None else "test_episode.mp4"
        video_path = self.trajectories_dir / video_filename
        
        # Save video using gymnasium's save_video utility
        save_video(video_frames, str(video_path), fps=fps)

    def plot_shortperiod_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        alpha_refs: np.ndarray,
        tracking_errors: np.ndarray,
        rewards: List[float],
        episode_num: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot trajectory specifically for short-period dynamics environment.

        Args:
            states: State trajectory array (timesteps, 2) - [alpha, q]
            actions: Action trajectory array (timesteps, 1) - [delta_e]
            alpha_refs: Reference alpha trajectory array (timesteps,)
            tracking_errors: Tracking error trajectory array (timesteps,)
            rewards: List of rewards per timestep
            episode_num: Optional episode number for labeling
            save_path: Path to save the plot
        """
        if not self.enabled:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=self.dpi)
        fig.patch.set_facecolor("black")
        axes = axes.flatten()

        timesteps = np.arange(len(states))
        dt = 0.01  # Assuming 0.01s timestep from config
        time = timesteps * dt

        # Plot 1: Alpha (angle of attack) with reference
        ax = axes[0]
        ax.plot(time, states[:, 0], color="cyan", linewidth=2, label="Alpha")
        ax.plot(time, alpha_refs, color="magenta", linestyle="--", linewidth=2, label="Alpha Reference")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Alpha (rad)")
        ax.legend(loc="best", facecolor="black", edgecolor="white", labelcolor="white")
        self._apply_dark_theme(ax)

        # Plot 2: Pitch rate (q)
        ax = axes[1]
        ax.plot(time, states[:, 1], color="cyan", linewidth=2, label="q (pitch rate)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("q (rad/s)")
        ax.legend(loc="best", facecolor="black", edgecolor="white", labelcolor="white")
        self._apply_dark_theme(ax)

        # Plot 3: Action (elevator deflection)
        ax = axes[2]
        ax.plot(time, actions[:, 0], color="yellow", linewidth=2, label="Elevator")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Delta_e (deg)")
        ax.legend(loc="best", facecolor="black", edgecolor="white", labelcolor="white")
        self._apply_dark_theme(ax)

        # Plot 4: Tracking error
        ax = axes[3]
        ax.plot(time, tracking_errors, color="red", linestyle="--", linewidth=2, label="Tracking Error")
        ax.axhline(y=0, color="white", linestyle=":", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Alpha Error (rad)")
        ax.legend(loc="best", facecolor="black", edgecolor="white", labelcolor="white")
        self._apply_dark_theme(ax)

        # Plot 5: Rewards
        ax = axes[4]
        ax.plot(time, rewards, color="green", linewidth=2, label="Reward")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Reward")
        ax.legend(loc="best", facecolor="black", edgecolor="white", labelcolor="white")
        self._apply_dark_theme(ax)

        # Plot 6: Cumulative reward
        ax = axes[5]
        cumulative_reward = np.cumsum(rewards)
        ax.plot(time, cumulative_reward, color="lime", linewidth=2, label="Cumulative Reward")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Reward")
        ax.legend(loc="best", facecolor="black", edgecolor="white", labelcolor="white")
        self._apply_dark_theme(ax)

        # Figure title
        ep_str = f"Episode {episode_num + 1}" if episode_num is not None else "Test Episode"
        total_reward = sum(rewards)
        fig.suptitle(
            f"{ep_str} — Short-Period Dynamics | Total Reward: {total_reward:.2f}",
            fontsize=14,
            color="white",
            y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save or show plot
        final_save_path = self._resolve_save_path(save_path, episode_num)
        if final_save_path:
            plt.savefig(final_save_path, facecolor="black", dpi=self.dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
