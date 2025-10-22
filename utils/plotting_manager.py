'''
Contains the plotting manager class. Plots are classified into Env and Agent plots.
'''
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class PlottingManager:
    def __init__(self, env: gym.Env, agent, save_dir: str = None):
        """
        Initialize PlottingManager.

        Args:
            env: Gymnasium environment
            agent: Agent instance
            save_dir: Directory to save all plots. If None, plots are displayed interactively.
        """
        import os

        self.env = env
        self.agent = agent
        self.save_dir = save_dir

        # Create save directory if specified
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def _get_save_path(self, filename: str) -> str:
        """Helper to construct save path for a plot."""
        import os
        return os.path.join(self.save_dir, filename) if self.save_dir else None
        
    def plot_pendulumcart_trajectory(self, states: np.ndarray, actions: np.ndarray,
                                      rewards: list[float] = None, filename: str = 'trajectory.png'):
        """
        Plot trajectory for pendulum cart environment.
        States: [x, x_dot, theta, theta_dot]
        Actions: [force]

        Args:
            states: State trajectory
            actions: Action trajectory
            rewards: Optional reward trajectory
            filename: Filename for saving (used only if save_dir was set in constructor)
        """
        dt = self.env.dt if hasattr(self.env, 'dt') else 0.02
        timesteps = np.arange(len(states))
        time = timesteps * dt

        # Subplot layout: 4 states + 1 action + 1 reward = 6 plots (3x2)
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Pendulum Cart Trajectory', fontsize=16, fontweight='bold')

        # State labels and units
        state_labels = ['Cart Position [m]', 'Cart Velocity [m/s]',
                       'Pendulum Angle [rad]', 'Angular Velocity [rad/s]']

        # Plot states
        for i in range(4):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            ax.plot(time, states[:, i], 'b-', linewidth=2)

            # Add reference line for angle at 0 (upright)
            if i == 2:  # theta
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Upright')

            PlotTools.apply_common_styling(ax, 'Time [s]', state_labels[i], state_labels[i],
                                          show_legend=(i == 2))

        # Plot action
        ax = axes[2, 0]
        ax.plot(time[:len(actions)], actions[:, 0], 'g-', linewidth=2)
        PlotTools.apply_common_styling(ax, 'Time [s]', 'Force [N]', 'Control Input', show_legend=False)

        # Plot rewards
        ax = axes[2, 1]
        if rewards is not None:
            ax.plot(time[:len(rewards)], rewards, 'r-', linewidth=2, alpha=0.7, label='Reward')
            cumulative_reward = np.cumsum(rewards)
            ax.plot(time[:len(rewards)], cumulative_reward, 'b-', linewidth=2, alpha=0.7, label='Cumulative')
            ax.set_title(f'Rewards (Total: {sum(rewards):.2f})')
            PlotTools.apply_common_styling(ax, 'Time [s]', 'Reward', f'Rewards (Total: {sum(rewards):.2f})')
        else:
            ax.axis('off')

        plt.tight_layout()
        PlotTools.save_or_show(fig, self._get_save_path(filename))
    
    def render_pendulumcart_env(self, states: np.ndarray, filename: str = 'animation.gif', fps: int = 30):
        """
        Render the pendulum cart environment as an animation.

        Args:
            states: Array of states [x, x_dot, theta, theta_dot] over time
            filename: Filename for animation (e.g., 'animation.gif' or 'animation.mp4')
            fps: Frames per second for rendering
        """
        from matplotlib.animation import FuncAnimation
        from matplotlib.patches import Rectangle, Circle

        dt = self.env.dt if hasattr(self.env, 'dt') else 0.02
        pendulum_length = self.env.l if hasattr(self.env, 'l') else 0.5

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 1.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Inverted Pendulum on Cart', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Draw track
        ax.plot([-3, 3], [-0.5, -0.5], 'k-', linewidth=2)

        # Initialize patches
        cart_width = 0.3
        cart_height = 0.2
        cart = Rectangle((0, -0.5), cart_width, cart_height,
                        facecolor='blue', edgecolor='black', linewidth=2)
        ax.add_patch(cart)

        # Pendulum line
        pendulum_line, = ax.plot([], [], 'r-', linewidth=3)

        # Pendulum bob
        bob = Circle((0, 0), 0.08, facecolor='red', edgecolor='black', linewidth=2)
        ax.add_patch(bob)

        # Info text
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        def init():
            """Initialize animation."""
            cart.set_xy((0, -0.5))
            pendulum_line.set_data([], [])
            bob.set_center((0, 0))
            info_text.set_text('')
            return cart, pendulum_line, bob, info_text

        def animate(frame):
            """Update animation frame."""
            if frame >= len(states):
                return cart, pendulum_line, bob, info_text

            x, _, theta, _ = states[frame]

            # Update cart position
            cart.set_xy((x - cart_width/2, -0.5))

            # Update pendulum
            pend_x = x + pendulum_length * np.sin(theta)
            pend_y = -0.5 + cart_height + pendulum_length * np.cos(theta)
            pendulum_line.set_data([x, pend_x], [-0.5 + cart_height, pend_y])
            bob.set_center((pend_x, pend_y))

            # Update info text
            info_text.set_text(f'x = {x:.3f} m\nθ = {theta:.3f} rad ({np.degrees(theta):.1f}°)\nStep = {frame}')

            return cart, pendulum_line, bob, info_text

        # Create animation
        interval = int(dt * 1000)  # Convert dt to milliseconds
        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=len(states), interval=interval,
                            blit=True, repeat=True)

        save_path = self._get_save_path(filename)
        if save_path:
            # Save animation
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps)
            else:
                anim.save(save_path, writer='ffmpeg', fps=fps)
            plt.close(fig)
            print(f"Animation saved to: {save_path}")
        else:
            plt.show()

        return anim
    
    def plot_hdp_learning(self, training_data: dict):
        """
        Plot comprehensive HDP learning metrics using all internal plotting methods.

        Args:
            training_data: Dictionary containing training metrics with keys:
                - 'critic_errors': List of critic TD errors
                - 'model_errors': List of model prediction errors
                - 'losses': Dict with 'actor', 'critic', 'model' keys
                - 'weight_norms': Dict with 'actor', 'critic', 'model' keys
                - 'weight_update_norms': Dict with 'actor', 'critic', 'model' keys
                - 'gradient_norms': Dict with 'actor', 'critic', 'model' keys
                - 'model_predictions': List of predicted states
                - 'true_states': List of actual states
        """
        # Plot critic TD errors
        if 'critic_errors' in training_data and training_data['critic_errors']:
            self._plot_error(training_data['critic_errors'], 'critic_errors.png')
            print(f"✓ Plotted critic errors{' (saved)' if self.save_dir else ''}")

        # Plot model prediction errors
        if 'model_errors' in training_data and training_data['model_errors']:
            self._plot_error(training_data['model_errors'], 'model_errors.png')
            print(f"✓ Plotted model errors{' (saved)' if self.save_dir else ''}")

        # Plot losses (actor, critic, model)
        if 'losses' in training_data and training_data['losses']:
            self._plot_losses(training_data['losses'], 'losses.png')
            print(f"✓ Plotted losses{' (saved)' if self.save_dir else ''}")

        # Plot weight norms
        if 'weight_norms' in training_data and training_data['weight_norms']:
            self._plot_weight_norms(training_data['weight_norms'], 'weight_norms.png')
            print(f"✓ Plotted weight norms{' (saved)' if self.save_dir else ''}")

        # Plot weight update norms
        if 'weight_update_norms' in training_data and training_data['weight_update_norms']:
            self._plot_weight_update_norms(training_data['weight_update_norms'], 'weight_update_norms.png')
            print(f"✓ Plotted weight update norms{' (saved)' if self.save_dir else ''}")

        # Plot gradient norms
        if 'gradient_norms' in training_data and training_data['gradient_norms']:
            self._plot_gradient_norms(training_data['gradient_norms'], 'gradient_norms.png')
            print(f"✓ Plotted gradient norms{' (saved)' if self.save_dir else ''}")

        # Plot model predictions vs actual states
        if ('model_predictions' in training_data and 'true_states' in training_data
            and training_data['model_predictions'] and training_data['true_states']):
            self._plot_model_predictions(training_data['model_predictions'],
                                        training_data['true_states'], 'model_predictions.png')
            print(f"✓ Plotted model predictions{' (saved)' if self.save_dir else ''}")

        print(f"\n{'='*50}")
        print("HDP Learning visualization complete!")
        if self.save_dir:
            print(f"All plots saved to: {self.save_dir}")
        print('='*50)
    
    def _plot_error(self, errors: list[float], filename: str):
        """Plot training errors (TD errors, model errors, etc.)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(errors, color='red', linewidth=1, alpha=0.7, label='Error')

        # Add moving average for better visualization
        if len(errors) > 50:
            window = 1
            ma = PlotTools.moving_average(errors, window)
            ax.plot(range(window-1, len(errors)), ma, color='blue', linewidth=2, label=f'MA({window})')

        PlotTools.apply_common_styling(ax, 'Training Step', 'Error', 'Training Error Over Time')
        PlotTools.save_or_show(fig, self._get_save_path(filename))
    
    def _plot_losses(self, losses: dict[str, list[float]], filename: str):
        """Plot multiple loss curves (actor, critic, model) on separate subplots."""
        n_losses = len(losses)
        if n_losses == 0:
            return

        fig, axes = plt.subplots(1, n_losses, figsize=(6*n_losses, 5))
        if n_losses == 1:
            axes = [axes]

        fig.suptitle('Training Losses', fontsize=14, fontweight='bold')

        for ax, (name, values) in zip(axes, losses.items()):
            ax.plot(values, linewidth=2)
            PlotTools.apply_common_styling(ax, 'Training Step', 'Loss',
                                          name.replace('_', ' ').title(), show_legend=False)

        plt.tight_layout()
        PlotTools.save_or_show(fig, self._get_save_path(filename))

    def _plot_weight_norms(self, weight_norms: dict[str, list[float]], filename: str):
        """Plot weight norms for each network (actor, critic, model) over training."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for network_name, norms in weight_norms.items():
            ax.plot(norms, linewidth=2, label=network_name, alpha=0.8)

        PlotTools.apply_common_styling(ax, 'Training Step', 'Weight Norm (L2)',
                                      'Network Weight Norms Over Training')
        PlotTools.save_or_show(fig, self._get_save_path(filename))

    def _plot_weight_update_norms(self, weight_update_norms: dict[str, list[float]], filename: str):
        """Plot weight update magnitudes for each network over training."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for network_name, norms in weight_update_norms.items():
            ax.plot(norms, linewidth=2, label=network_name, alpha=0.8)

        ax.set_yscale('log')  # Often useful for update magnitudes
        PlotTools.apply_common_styling(ax, 'Training Step', 'Weight Update Norm (L2)',
                                      'Weight Update Magnitudes Over Training')
        PlotTools.save_or_show(fig, self._get_save_path(filename))

    def _plot_gradient_norms(self, gradient_norms: dict[str, list[float]], filename: str):
        """Plot gradient norms for each network over training."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for network_name, norms in gradient_norms.items():
            ax.plot(norms, linewidth=2, label=network_name, alpha=0.8)

        ax.set_yscale('log')  # Often useful for gradient magnitudes
        PlotTools.apply_common_styling(ax, 'Training Step', 'Gradient Norm (L2)',
                                      'Gradient Norms Over Training')
        PlotTools.save_or_show(fig, self._get_save_path(filename))

    def _plot_model_predictions(self, model_predictions: list[np.ndarray],
                                true_next_states: list[np.ndarray], filename: str):
        """
        Plot model predictions vs actual states for each state dimension.

        Args:
            model_predictions: List of predicted next states
            true_next_states: List of actual next states
        """
        # Convert to arrays
        predictions = PlotTools.ensure_array(model_predictions)
        true_states = PlotTools.ensure_array(true_next_states)

        state_dim = predictions.shape[1] if len(predictions.shape) > 1 else 1
        timesteps = np.arange(len(predictions))

        # Determine subplot layout
        rows, cols = PlotTools.calculate_subplot_layout(state_dim)

        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        fig.suptitle('Model Predictions vs Actual States', fontsize=14, fontweight='bold')

        # Flatten axes for easier iteration
        if state_dim == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else axes

        # Plot each state dimension
        for i in range(state_dim):
            ax = axes[i]
            ax.plot(timesteps, true_states[:, i], 'b-', linewidth=2, label='Actual', alpha=0.8)
            ax.plot(timesteps, predictions[:, i], 'r--', linewidth=2, label='Predicted', alpha=0.8)
            PlotTools.apply_common_styling(ax, 'Timestep', f'State {i}', f'State Dimension {i}')

        # Hide unused subplots
        for i in range(state_dim, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        PlotTools.save_or_show(fig, self._get_save_path(filename))
    
    
class PlotTools:
    ''' Helper class for the PlottingManager to keep PlottingManager clean. '''

    @staticmethod
    def moving_average(data: list[float], window_size: int) -> np.ndarray:
        """
        Compute moving average of data with given window size.

        Args:
            data: List of values to smooth
            window_size: Window size for moving average

        Returns:
            Array of smoothed values (length will be len(data) - window_size + 1)
        """
        if len(data) < window_size:
            return np.array(data)
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    @staticmethod
    def apply_common_styling(ax: plt.Axes, xlabel: str, ylabel: str, title: str, show_legend: bool = True):
        """
        Apply common styling to a matplotlib axis.

        Args:
            ax: Matplotlib axis object
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            title: Title for the plot
            show_legend: Whether to show legend
        """
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend()

    @staticmethod
    def save_or_show(fig: plt.Figure, save_path: str = None, dpi: int = 150):
        """
        Save figure to file or show it.

        Args:
            fig: Matplotlib figure object
            save_path: Path to save figure (if None, will show instead)
            dpi: DPI for saved figure
        """
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def calculate_subplot_layout(n_plots: int, max_cols: int = 3) -> tuple[int, int]:
        """
        Calculate optimal subplot layout for given number of plots.

        Args:
            n_plots: Number of subplots needed
            max_cols: Maximum number of columns

        Returns:
            Tuple of (rows, cols)
        """
        if n_plots <= 2:
            return 1, n_plots
        elif n_plots <= 4:
            return 2, 2
        else:
            cols = min(n_plots, max_cols)
            rows = (n_plots + cols - 1) // cols
            return rows, cols

    @staticmethod
    def ensure_array(data: list[np.ndarray]) -> np.ndarray:
        """
        Convert list of arrays to numpy array.

        Args:
            data: List of numpy arrays or list

        Returns:
            Numpy array
        """
        return np.array(data)