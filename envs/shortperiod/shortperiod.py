"""
Short-Period Pitch Dynamics Environment with Sinusoidal Reference Tracking

This module implements a Gymnasium environment for the linearized short-period
dynamics of an air vehicle. The agent must track a sinusoidal angle-of-attack
reference signal by controlling the elevator deflection.

State vector: x = [alpha, q] where
    - alpha: angle of attack [rad]
    - q: pitch rate [rad/s]

Control input: u = [delta_e] (elevator deflection [deg])

Dynamics: x_dot = A*x + B*u

Reference: alpha_ref(t) = A_ref * sin(2*pi*t / T_ref)

Observation: [alpha, q, alpha_ref]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class ShortPeriodEnv(gym.Env):

    def __init__(
        self,
        dt: float = 0.01,
        max_steps: int = 1000
    ):
        super().__init__()

        # System matrices
        self.A = np.array([[-0.6761,  0.9392], [-0.5757, -0.8741]])
        self.B = np.array([[-0.001437], [-0.1188]])
         
        # Reward function parameters
        self.A_ref = 0.2  # rad (~11.5 degrees)
        self.T_ref = 2.0  # seconds

        # Episode parameters
        self.max_steps = max_steps

        # SPACES
        self.alpha_max = 0.4*2  # rad (~20 degrees)
        self.q_max = 1.5       # rad/s (~86 deg/s)
        self.delta_max = 25.0  # deg
        
        # Define state space: [alpha, q]
        self.state_space = spaces.Box(
            low=np.array([-self.alpha_max, -self.q_max]),
            high=np.array([self.alpha_max, self.q_max]),
            dtype=np.float32
        )

        # Define observation space: [alpha, q, alpha_ref]
        self.obs_space = spaces.Box(
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
        self.alpha_ref = None
        self.time = 0.0
        self.step_count = 0
        self.dt = dt

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

        # Initialize state near trim condition with small perturbation
        self.state = np.random.normal(0.0, 0.01, size=2)
        self.time = 0.0
        self.step_count = 0

        # Compute initial observation
        self.alpha_ref = self._get_reference(self.time)
        observation = np.array([self.state[0], self.state[1], self.alpha_ref], dtype=np.float32)
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
        u = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        delta_e = u[0]
        x = self.state

        # Get current reference
        self.alpha_ref = self._get_reference(self.time)

    
        xdot = self._dynamics(x, u)
        xnext = x + xdot.flatten() * self.dt
        self.state = xnext

        # Update time
        self.time += self.dt
        self.step_count += 1

        # Compute reward
        alpha = self.state[0]
        q = self.state[1]
        tracking_error = self.alpha_ref - alpha

        reward = -tracking_error**2

        # Check termination conditions
        terminated = False
        if not self.state_space.contains(self.state.astype(np.float32)):
            terminated = True


        # Check truncation (time limit)
        truncated = self.step_count >= self.max_steps

        # Compute new observation
        observation = np.array([self.state[0], self.state[1], self.alpha_ref], dtype=np.float32)

        # Additional info
        info = {
            "time": self.time,
            "alpha": alpha,
            "q": q,
            "alpha_ref": self.alpha_ref,
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
            alpha_ref: Reference angle of attack [rad]
        """
        return self.A_ref * np.sin(2.0 * np.pi * t / self.T_ref)

    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
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
        return x_dot


