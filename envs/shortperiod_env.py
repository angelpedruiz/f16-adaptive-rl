"""
Short-Period Pitch Dynamics Environment with Sinusoidal Reference Tracking

This module implements a Gymnasium environment for the linearized short-period
dynamics of an air vehicle. The agent must track a sinusoidal angle-of-attack
reference signal by controlling the elevator deflection.

State vector: x = [alpha, q]^T where
    - alpha: angle of attack [rad]
    - q: pitch rate [rad/s]

Control input: u = delta_e (elevator deflection [rad])

Dynamics: x_dot = A*x + B*u + w(t)
          y = C*x

Reference: alpha_ref(t) = A_ref * sin(2*pi*t / T_ref)

Observation: [alpha, q, alpha_ref]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
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

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        dt: float = 0.02,
        A_ref: float = 0.1,
        T_ref: float = 10.0,
        process_noise_std: float = 0.0,
        w_alpha: float = 100.0,
        w_q: float = 10.0,
        w_u: float = 1.0,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
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
            render_mode: Rendering mode (only "human" is supported)
        """
        super().__init__()

        # System matrices
        self.A = A if A is not None else np.array([[-0.5, 1.0], [-20.0, -2.0]])
        self.B = B if B is not None else np.array([[0.0], [5.0]])

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

        # State limits
        self.alpha_max = 0.35  # rad (~20 degrees)
        self.q_max = 50.0      # rad/s
        self.delta_max = np.deg2rad(25.0)  # rad

        # Termination limits (slightly larger than observation limits)
        self.alpha_term = 0.4   # rad
        self.q_term = 200.0     # rad/s
        self.term_penalty = -1000.0

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

        # Rendering
        self.render_mode = render_mode
        self.history = {
            "time": [],
            "alpha": [],
            "alpha_ref": [],
            "q": [],
            "delta_e": [],
        }

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

        # Reset history
        self.history = {
            "time": [],
            "alpha": [],
            "alpha_ref": [],
            "q": [],
            "delta_e": [],
        }

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
            action: Control input [delta_e] in radians

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

        # Store history for rendering
        self.history["time"].append(self.time)
        self.history["alpha"].append(self.state[0])
        self.history["alpha_ref"].append(alpha_ref)
        self.history["q"].append(self.state[1])
        self.history["delta_e"].append(delta_e)

        # Compute reward
        alpha = self.state[0]
        q = self.state[1]
        tracking_error = alpha - alpha_ref

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

    def render(self):
        """
        Render the environment state using matplotlib.

        Displays time history plots of:
        - Angle of attack (alpha) and reference (alpha_ref)
        - Pitch rate (q)
        - Elevator deflection (delta_e)
        """
        if len(self.history["time"]) == 0:
            print("No data to render. Run at least one step first.")
            return

        time = np.array(self.history["time"])
        alpha = np.rad2deg(np.array(self.history["alpha"]))
        alpha_ref = np.rad2deg(np.array(self.history["alpha_ref"]))
        q = np.rad2deg(np.array(self.history["q"]))
        delta_e = np.rad2deg(np.array(self.history["delta_e"]))

        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        # Plot angle of attack
        axes[0].plot(time, alpha, label=r"$\alpha$ (actual)", linewidth=2)
        axes[0].plot(time, alpha_ref, label=r"$\alpha_{ref}$ (reference)",
                     linestyle="--", linewidth=2)
        axes[0].set_ylabel("Angle of Attack [deg]")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Short-Period Pitch Dynamics - Reference Tracking")

        # Plot pitch rate
        axes[1].plot(time, q, label="q (pitch rate)", linewidth=2, color="orange")
        axes[1].set_ylabel("Pitch Rate [deg/s]")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot elevator deflection
        axes[2].plot(time, delta_e, label=r"$\delta_e$ (elevator)", linewidth=2, color="green")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("Elevator Deflection [deg]")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def close(self):
        """Clean up resources."""
        plt.close("all")

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
            u: Control input delta_e [rad]

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
            x: Current state [�, q]
            u: Control input delta_e [rad]
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


