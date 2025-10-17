"""
Short-Period Pitch Dynamics Environment with Sinusoidal Reference Tracking

This module implements a Gymnasium environment for the linearized short-period
dynamics of an air vehicle. The agent must track a sinusoidal angle-of-attack
reference signal by controlling the elevator deflection.

State vector: x = [alpha, q]^T where
    - alpha: angle of attack [rad]
    - q: pitch rate [rad/s]

Control input: u = delta_e (elevator deflection [deg])

Dynamics: x_dot = A*x + B*u + w(t)
          y = C*x

Reference: alpha_ref(t) = A_ref * sin(2*pi*t / T_ref)

Observation: [alpha, q, alpha_ref]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class ShortPeriodEnv(gym.Env):
    """
    Gymnasium environment for linearized short-period pitch dynamics with
    sinusoidal reference tracking.

    The objective is to track a time-varying angle-of-attack reference signal
    alpha_ref(t) = A_ref * sin(2*pi*t / T_ref) by applying elevator deflections.

    Attributes:
        A (np.ndarray): System dynamics matrix (2x2)
        B (np.ndarray): Control input matrix (2x1)
        C (np.ndarray): Output matrix (2x2)
        dt (float): Integration time step [s]
        A_ref (float): Reference amplitude [rad]
        T_ref (float): Reference period [s]
        process_noise_std (float): Standard deviation of process noise
        w_alpha (float): Tracking error weight in reward
        w_q (float): Pitch rate penalty weight in reward
        w_u (float): Control effort penalty weight in reward
        max_episode_steps (int): Maximum steps per episode
    """

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        dt: float = 0.02,
        A_ref: float = 0.1,
        T_ref: float = 10.0,
        process_noise_std: float = 0.0,
        w_alpha: float = 1.0,
        w_q: float = 0.0,
        w_u: float = 0.0,
        max_episode_steps: int = 1000
    ):
        """
        Initialize the short-period pitch dynamics environment.

        Args:
            A: System dynamics matrix (2x2). Default: [[-0.5, 1.0], [-20.0, -2.0]]
            B: Control input matrix (2x1). Default: [[0.0], [5.0]]
            C: Output matrix (2x2). Default: identity
            dt: Integration time step [s]
            A_ref: Reference signal amplitude [rad]
            T_ref: Reference signal period [s]
            process_noise_std: Standard deviation of additive Gaussian noise
            w_alpha: Weight for tracking error in reward function
            w_q: Weight for pitch rate penalty in reward function
            w_u: Weight for control effort penalty in reward function
            max_episode_steps: Maximum number of steps per episode
        """
        super().__init__()

        # System matrices
        self.A = A if A is not None else np.array([[-0.6761,  0.9392], [-0.5757, -0.8741]])
        self.B = B if B is not None else np.array([[-0.001437], [-0.1188]])

        # Simulation parameters
        self.dt = dt
        self.A_ref = A_ref
        self.T_ref = T_ref
        self.process_noise_std = process_noise_std

        # Reward function weights
        self.w_alpha = w_alpha
        self.w_q = w_q
        self.w_u = w_u

        # Episode parameters
        self.max_episode_steps = max_episode_steps

        # State limits (realistic values for aircraft dynamics)
        self.alpha_max = 0.4*2  # rad (~20 degrees)
        self.q_max = 1.5       # rad/s (~86 deg/s)
        self.delta_max = 25.0  # deg

        # Termination limits (slightly larger than observation limits)
        self.alpha_term = 0.52*2   # rad (~30 degrees)
        self.q_term = 2.0        # rad/s (~115 deg/s)
        self.term_penalty = -0.0

        # Define observation space: [alpha, q, alpha_ref]
        self.observation_space = spaces.Box(
            low=np.array([-self.alpha_max, -self.q_max, -self.alpha_max]),
            high=np.array([self.alpha_max, self.q_max, self.alpha_max]),
            dtype=np.float32,
        )

        # Define action space: [delta_e]
        self.action_space = spaces.Box(
            low=np.array([-self.delta_max]),
            high=np.array([self.delta_max]),
            dtype=np.float32,
        )

        # Internal state
        self.state = None  # [alpha, q]
        self.time = 0.0
        self.step_count = 0
        self.rng = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial conditions.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation [alpha, q, alpha_ref]
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Initialize random number generator
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        # Initialize state near trim condition with small perturbation
        self.state = self.rng.normal(0.0, 0.01, size=2)
        self.time = 0.0
        self.step_count = 0

        # Compute initial observation
        alpha_ref = self._get_reference(self.time)
        observation = np.array([self.state[0], self.state[1], alpha_ref], dtype=np.float32)

        info = {}

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.

        Args:
            action: Control input [delta_e] in degrees

        Returns:
            observation: New observation [alpha, q, alpha_ref]
            reward: Reward for this step
            terminated: Whether episode ended due to constraint violation
            truncated: Whether episode ended due to time limit
            info: Additional information dictionary
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_e = float(action[0])

        # Get current reference
        alpha_ref = self._get_reference(self.time)

        # Integrate dynamics using RK4
        self.state = self._rk4_step(self.state, delta_e, self.dt)

        # Update time
        self.time += self.dt
        self.step_count += 1

        # Compute reward
        alpha = self.state[0]
        q = self.state[1]
        tracking_error = alpha_ref - alpha

        reward = -(
            self.w_alpha * tracking_error**2
            + self.w_q * q**2
            + self.w_u * delta_e**2
        )

        # Check termination conditions
        terminated = False
        if abs(alpha) > self.alpha_term or abs(q) > self.q_term:
            terminated = True
            reward += self.term_penalty

        # Check truncation (time limit)
        truncated = self.step_count >= self.max_episode_steps

        # Compute new observation
        alpha_ref_new = self._get_reference(self.time)
        observation = np.array([self.state[0], self.state[1], alpha_ref_new], dtype=np.float32)

        # Additional info
        info = {
            "time": self.time,
            "alpha": alpha,
            "q": q,
            "alpha_ref": alpha_ref_new,
            "delta_e": delta_e,
            "tracking_error": tracking_error,
        }

        return observation, float(reward), terminated, truncated, info

    def _get_reference(self, t: float) -> float:
        """
        Compute the reference angle of attack at time t.

        Args:
            t: Current time [s]

        Returns:
            �_ref: Reference angle of attack [rad]
        """
        return self.A_ref * np.sin(2.0 * np.pi * t / self.T_ref)

    def _dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        """
        Compute state derivatives with optional process noise.

        Args:
            x: State vector [alpha, q]
            u: Control input delta_e [deg]

        Returns:
            x_dot: State derivative [alpha_dot, q_dot]
        """
        # Compute deterministic dynamics: x_dot = A*x + B*u
        x_dot = self.A @ x + self.B.flatten() * u

        # Add process noise if specified
        if self.process_noise_std > 0:
            noise = self.rng.normal(0.0, self.process_noise_std, size=2)
            x_dot += noise

        return x_dot

    def _rk4_step(self, x: np.ndarray, u: float, dt: float) -> np.ndarray:
        """
        Perform one RK4 integration step.

        Args:
            x: Current state [alpha, q]
            u: Control input delta_e [deg]
            dt: Time step [s]

        Returns:
            x_next: Next state after dt seconds
        """
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + 0.5 * dt * k1, u)
        k3 = self._dynamics(x + 0.5 * dt * k2, u)
        k4 = self._dynamics(x + dt * k3, u)

        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return x_next


if __name__ == "__main__":
    """Minimal training loop for testing ADHDP with ShortPeriodEnv."""
    import sys
    from pathlib import Path
    import matplotlib.pyplot as plt
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from agents.adhdp import ADHDPAgent

    # Create environment
    env = ShortPeriodEnv(
        dt=0.01,
        A_ref=0.35,
        T_ref=5.0,
        w_alpha=1.0,
        w_q=0.0,
        w_u=0.0,
        max_episode_steps=1000  # Reduced for faster testing
    )

    # Create agent
    agent = ADHDPAgent(
        obs_dim=3,
        act_dim=1,
        hidden_sizes=[32, 32],
        actor_lr=0.01,
        critic_lr=0.01,
        gamma=0.8,
        device='cuda',
        action_low=env.action_space.low,
        action_high=env.action_space.high
    )

    # Training loop with trajectory recording
    num_episodes = 1
    max_steps = 1000  # Match env max_steps
    print(f"Training ADHDP on ShortPeriodEnv for {num_episodes} episodes")
    print("=" * 60)

    # Storage for trajectories (padded to max_steps with NaN)
    all_alphas = np.full((num_episodes, max_steps), np.nan)
    all_qs = np.full((num_episodes, max_steps), np.nan)
    all_alpha_refs = np.full((num_episodes, max_steps), np.nan)
    all_delta_es = np.full((num_episodes, max_steps), np.nan)
    all_rewards = np.full((num_episodes, max_steps), np.nan)
    episode_lengths = []

    for episode in range(num_episodes):
        # Reset agent for online learning (no memory between episodes)
            # Create agent
        agent = ADHDPAgent(
            obs_dim=3,
            act_dim=1,
            hidden_sizes=[32, 32],
            actor_lr=0.01,
            critic_lr=0.01,
            gamma=0.8,
            device='cuda',
            action_low=env.action_space.low,
            action_high=env.action_space.high
        )

        obs, _ = env.reset(seed=episode)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Get action
            action = agent.get_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Record trajectory
            all_alphas[episode, steps] = info['alpha']
            all_qs[episode, steps] = info['q']
            all_alpha_refs[episode, steps] = info['alpha_ref']
            all_delta_es[episode, steps] = info['delta_e']
            all_rewards[episode, steps] = reward

            # Update agent
            agent.update(obs, action, reward, terminated, next_obs)

            total_reward += reward
            steps += 1
            obs = next_obs

        episode_lengths.append(steps)

        # Print episode summary
        status = "TERMINATED" if terminated else "TRUNCATED"
        print(f"\nEpisode {episode+1:3d}: {status:10s} | Steps: {steps:4d} | Total Reward: {total_reward:8.2f}")
        print(f"  Final alpha: {info['alpha']:6.3f} rad | Final q: {info['q']:6.3f} rad/s")
        print("-" * 60)

    print("=" * 60)
    print("Training complete!")

    # Plot mean trajectory with variance
    print("\nGenerating trajectory plot...")

    # Compute statistics (ignoring NaN values)
    mean_alpha = np.nanmean(all_alphas, axis=0)
    std_alpha = np.nanstd(all_alphas, axis=0)
    mean_q = np.nanmean(all_qs, axis=0)
    std_q = np.nanstd(all_qs, axis=0)
    mean_alpha_ref = np.nanmean(all_alpha_refs, axis=0)
    mean_delta_e = np.nanmean(all_delta_es, axis=0)
    std_delta_e = np.nanstd(all_delta_es, axis=0)
    mean_reward = np.nanmean(all_rewards, axis=0)
    std_reward = np.nanstd(all_rewards, axis=0)

    # Compute tracking error statistics
    mean_tracking_error = mean_alpha_ref - mean_alpha
    std_tracking_error = std_alpha  # Error std is same as alpha std

    # Time vector
    time = np.arange(max_steps) * env.dt

    # Create figure with dark theme (similar to plotting manager)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    fig.patch.set_facecolor("black")
    axes = axes.flatten()

    # Helper function to apply dark theme
    def apply_dark_theme(ax):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.grid(True, alpha=0.3, color="white")
        for spine in ax.spines.values():
            spine.set_color("white")

    # Plot 1: Alpha tracking with reference
    ax = axes[0]
    ax.plot(time, mean_alpha, color='cyan', linewidth=2, label='Mean α')
    ax.fill_between(time,
                     mean_alpha - std_alpha,
                     mean_alpha + std_alpha,
                     alpha=0.3, color='cyan', label='±1 std')
    ax.plot(time, mean_alpha_ref, color='magenta', linestyle='--', linewidth=2, label='Reference α')
    ax.axhline(env.alpha_term, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(-env.alpha_term, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Alpha [rad]')
    ax.legend(loc='best', facecolor='black', edgecolor='white', labelcolor='white')
    apply_dark_theme(ax)

    # Plot 2: Pitch rate
    ax = axes[1]
    ax.plot(time, mean_q, color='cyan', linewidth=2, label='Mean q')
    ax.fill_between(time, mean_q - std_q, mean_q + std_q, alpha=0.3, color='cyan', label='±1 std')
    ax.axhline(env.q_term, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(-env.q_term, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('q [rad/s]')
    ax.legend(loc='best', facecolor='black', edgecolor='white', labelcolor='white')
    apply_dark_theme(ax)

    # Plot 3: Elevator deflection (action)
    ax = axes[2]
    ax.plot(time, mean_delta_e, color='yellow', linewidth=2, label='Mean δₑ')
    ax.fill_between(time, mean_delta_e - std_delta_e, mean_delta_e + std_delta_e,
                     alpha=0.3, color='yellow', label='±1 std')
    ax.axhline(env.delta_max, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(-env.delta_max, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Delta_e [deg]')
    ax.legend(loc='best', facecolor='black', edgecolor='white', labelcolor='white')
    apply_dark_theme(ax)

    # Plot 4: Tracking error
    ax = axes[3]
    ax.plot(time, mean_tracking_error, color='red', linestyle='--', linewidth=2, label='Mean Error')
    ax.fill_between(time,
                     mean_tracking_error - std_tracking_error,
                     mean_tracking_error + std_tracking_error,
                     alpha=0.3, color='red', label='±1 std')
    ax.axhline(0, color='white', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Alpha Error [rad]')
    ax.legend(loc='best', facecolor='black', edgecolor='white', labelcolor='white')
    apply_dark_theme(ax)

    # Plot 5: Reward
    ax = axes[4]
    ax.plot(time, mean_reward, color='green', linewidth=2, label='Mean Reward')
    ax.fill_between(time, mean_reward - std_reward, mean_reward + std_reward,
                     alpha=0.3, color='green', label='±1 std')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Reward')
    ax.legend(loc='best', facecolor='black', edgecolor='white', labelcolor='white')
    apply_dark_theme(ax)

    # Plot 6: Cumulative reward
    ax = axes[5]
    cumulative_reward = np.nancumsum(mean_reward)
    ax.plot(time, cumulative_reward, color='lime', linewidth=2, label='Cumulative Reward')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Cumulative Reward')
    ax.legend(loc='best', facecolor='black', edgecolor='white', labelcolor='white')
    apply_dark_theme(ax)

    # Figure title
    total_reward = np.nansum(mean_reward)
    fig.suptitle(
        f'ADHDP on Short-Period Environment | {num_episodes} Episodes | Mean Total Reward: {total_reward:.2f}',
        fontsize=14,
        color='white',
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()
    plt.show()

    print(f"\nMean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print("Close the plot window to exit.")
