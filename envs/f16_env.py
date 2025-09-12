import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from data.LinearF16SS import B_f1
from utils.reference_utils import cosine_smooth, step_reference, generate_cos_step_sequence


class LinearModelF16(gym.Env):
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        max_steps: int = 3000,
        dt: float = 0.01,
        reference_config: Optional[dict] = None,
        state_indices_for_obs: Optional[list] = None,
        obs_low: Optional[list] = None,
        obs_high: Optional[list] = None,
        action_low: Optional[list] = None,
        action_high: Optional[list] = None,
    ):
        # Convert matrices to numpy arrays and ensure proper dtypes
        self.A = np.array(A, dtype=np.float64)
        self.B = np.array(B, dtype=np.float64)
        self.state_dim = self.A.shape[1]
        self.action_dim = self.B.shape[1]

        self.max_steps = max_steps
        self.dt = dt

        # Convert bounds to numpy arrays with proper validation
        if obs_low is None:
            obs_low = [-np.inf] * len(
                state_indices_for_obs or list(range(self.state_dim))
            )
        if obs_high is None:
            obs_high = [np.inf] * len(
                state_indices_for_obs or list(range(self.state_dim))
            )
        if action_low is None:
            action_low = [-np.inf] * self.action_dim
        if action_high is None:
            action_high = [np.inf] * self.action_dim

        self.obs_low = np.array(obs_low, dtype=np.float64)
        self.obs_high = np.array(obs_high, dtype=np.float64)
        self.action_low = np.array(action_low, dtype=np.float64)
        self.action_high = np.array(action_high, dtype=np.float64)

        # Define action space
        self.action_space = spaces.Box(
            low=self.action_low,
            high=self.action_high,
            shape=(self.action_dim,),
            dtype=np.float64,
        )

        # Determine observation structure
        self.state_indices_for_obs = state_indices_for_obs or list(range(self.state_dim))
        self.reference_config = reference_config or {}
        self.tracking_indices = list(self.reference_config.keys()) if self.reference_config else []
        
        # Calculate actual observation dimensions
        self.n_state_obs = len(self.state_indices_for_obs)
        self.n_error_obs = len(self.tracking_indices)
        self.total_obs_dim = self.n_state_obs + self.n_error_obs
        
        # Validate and construct observation bounds
        if len(self.obs_low) != self.total_obs_dim or len(self.obs_high) != self.total_obs_dim:
            raise ValueError(
                f"Observation bounds length mismatch: expected {self.total_obs_dim} elements "
                f"({self.n_state_obs} state + {self.n_error_obs} error components), "
                f"got {len(self.obs_low)} low bounds and {len(self.obs_high)} high bounds"
            )

        # Define observation space with validated dimensions
        self.observation_space = spaces.Box(
            low=self.obs_low,
            high=self.obs_high,
            shape=(self.total_obs_dim,),
            dtype=np.float64,
        )

        # Initialize state and tracking variables
        self.state = np.zeros(self.state_dim, dtype=np.float64)
        self.current_step = 0
        self.terminated = False
        self.prev_action = None

    def _get_obs(self) -> np.ndarray:
        """
        Get current observation as numpy array with guaranteed shape consistency.

        Returns:
            np.ndarray: Current observation containing selected states and tracking errors.
        """
        # Pre-allocate observation array for optimal performance
        observation = np.empty(self.total_obs_dim, dtype=np.float64)
        
        # Fill state components (vectorized indexing)
        observation[:self.n_state_obs] = self.state[self.state_indices_for_obs]
        
        # Fill tracking errors if present
        if self.n_error_obs > 0:
            # Use cached reference if available, otherwise compute it
            ref = getattr(self, '_cached_reference', None)
            if ref is None:
                ref = self._get_reference()
            observation[self.n_state_obs:] = ref[self.tracking_indices] - self.state[self.tracking_indices]
        
        return observation

    def _get_info(self) -> dict:
        """
        Get environment information dictionary.

        Returns:
            dict: Information about current environment state.
        """
        # Use cached reference if available, otherwise compute it
        ref = getattr(self, '_cached_reference', None)
        if ref is None:
            ref = self._get_reference()
        # Vectorized tracking error calculation using pre-computed indices
        if self.n_error_obs > 0:
            tracking_error = ref[self.tracking_indices] - self.state[self.tracking_indices]
        else:
            tracking_error = np.array([], dtype=np.float64)

        info = {
            "time_s": self.current_step * self.dt,
            "state": self.state.copy(),
            "observation": self._get_obs(),
            "reference": ref,  # CRITICAL: Keep this for plotting compatibility
            "tracking_error": tracking_error,
            "tracking_mse": np.mean(tracking_error**2)
            if len(tracking_error) > 0
            else 0.0,
            "reward": self._get_reward(
                self.state, self.prev_action, self.terminated, False
            ),
        }

        if self.prev_action is not None:
            info["last_action"] = self.prev_action.copy()

        return info

    def _get_info_optimized(self, observation: np.ndarray, reward: float) -> dict:
        """
        Get environment information dictionary (optimized to avoid redundant calculations).
        
        Args:
            observation: Pre-computed observation
            reward: Pre-computed reward
            
        Returns:
            dict: Information about current environment state.
        """
        # Use cached reference (should already be available from step)
        ref = getattr(self, '_cached_reference', self._get_reference())
        
        # Vectorized tracking error calculation using pre-computed indices
        if self.n_error_obs > 0:
            tracking_error = ref[self.tracking_indices] - self.state[self.tracking_indices]
        else:
            tracking_error = np.array([], dtype=np.float64)

        info = {
            "time_s": self.current_step * self.dt,
            "state": self.state,  # Remove unnecessary .copy()
            "observation": observation,  # Use pre-computed observation
            "reference": ref,  # CRITICAL: Keep this for plotting compatibility
            "tracking_error": tracking_error,
            "tracking_mse": np.mean(tracking_error**2) if len(tracking_error) > 0 else 0.0,
            "reward": reward,  # Use pre-computed reward
        }
        
        # Include last action if available
        if self.prev_action is not None:
            info["last_action"] = self.prev_action  # Remove .copy() for speed

        return info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.state = np.zeros(self.state_dim, dtype=np.float64)
        self.current_step = 0
        self.terminated = False
        self.prev_action = None

        if options:
            fault_type = options.get('fault_type', 'null')
            if fault_type == "elevator_loss":
                self.B = B_f1
            elif fault_type is None:
                pass
            else:
                print(f'env.reset() error: fault {fault_type} not recognised')

        # --- generate references *before* first observation
        # Precompute cos_step references as matrix for efficiency
        self.cos_step_matrix = None
        if self.reference_config:
            self.reference = {}
            cos_step_indices = []
            cos_step_sequences = []
            
            for idx, cfg in self.reference_config.items():
                if cfg["type"] == "cos_step":
                    time_seq, ref_seq = generate_cos_step_sequence(
                        cfg,
                        max_time=self.max_steps * self.dt,
                        dt=self.dt,
                        seed=cfg.get("seed", None),
                    )
                    # Store for backward compatibility with existing code
                    self.reference[idx] = {"t": time_seq, "y": ref_seq}
                    cos_step_indices.append(idx)
                    cos_step_sequences.append(ref_seq)
                else:
                    self.reference[idx] = None
            
            # Create precomputed matrix for cos_step references
            if cos_step_indices:
                # Initialize matrix with None (object dtype)
                self.cos_step_matrix = np.full((self.state_dim, self.max_steps), None, dtype=object)
                # Fill in the cos_step sequences for tracked states
                for i, state_idx in enumerate(cos_step_indices):
                    # Ensure we don't exceed matrix bounds
                    seq_len = min(len(cos_step_sequences[i]), self.max_steps)
                    self.cos_step_matrix[state_idx, :seq_len] = cos_step_sequences[i][:seq_len]
        else:
            self.reference = {}

        # Now safe to call _get_obs (uses self.reference)
        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def _get_reference(self) -> np.ndarray:
        """
        Compute reference signals at current time.

        Supports:
            - sin: sinusoidal reference
            - constant: constant value
            - cos_step: cosine-smoothed step function with discrete amplitude levels (precomputed)

        Returns:
            np.ndarray: Reference values for all states. 
                        Untracked states are None.
        """
        ref = np.full(self.state_dim, None, dtype=object)  # allow None entries
        t = self.current_step * self.dt

        for idx, cfg in self.reference_config.items():
            if cfg["type"] == "sin":
                omega = 2 * np.pi / cfg["T"]
                ref[idx] = cfg["A"] * np.sin(omega * t + cfg.get("phi", 0.0))

            elif cfg["type"] == "constant":
                ref[idx] = cfg["value"]

            elif cfg["type"] == "cos_step":
                # Use precomputed matrix for efficiency
                if self.cos_step_matrix is not None and self.current_step < self.cos_step_matrix.shape[1]:
                    ref[idx] = self.cos_step_matrix[idx, self.current_step]
                else:
                    # Fallback to original method if matrix not available
                    seq_t = self.reference[idx]["t"]
                    seq_y = self.reference[idx]["y"]
                    time_idx = min(int(t / self.dt), len(seq_t) - 1)
                    ref[idx] = seq_y[time_idx]

            else:
                raise ValueError(f"Unknown reference type '{cfg['type']}' for state {idx}")

        return ref



    def _get_reward(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray],
        terminated: bool,
        truncated: bool,
    ) -> float:
        """
        Compute reward based on tracking performance.

        Args:
            state: Current state vector.
            action: Current action vector (unused in basic reward).
            terminated: Whether episode terminated early.
            truncated: Whether episode reached max steps.

        Returns:
            float: Computed reward value.
        """
        # Use cached reference if available, otherwise compute it
        ref = getattr(self, '_cached_reference', None)
        if ref is None:
            ref = self._get_reference()

        if self.n_error_obs > 0:
            # Only compute tracking error for states with non-None references
            tracking_error = []
            for idx in self.tracking_indices:
                if ref[idx] is not None:
                    tracking_error.append(ref[idx] - state[idx])
            tracking_error = np.array(tracking_error, dtype=np.float64)
            squared_error = np.sum(tracking_error**2)

            # Define max allowable error based on reasonable bounds for tracking errors
            # Use a simple heuristic: sum of squared ranges for each tracked state
            Emax = 0
            for i, state_idx in enumerate(self.tracking_indices):
                # Use a reasonable error bound (e.g., ±2.0 for most tracking tasks)
                # This could be made configurable in the future
                error_range = 4.0  # Assuming ±2.0 error range per tracked state
                Emax += error_range ** 2

            # Shifted reward clipped at 0
            reward = max(0.0, Emax - squared_error)
        else:
            # No reference config means no tracking; reward zero
            reward = 0.0

        # Penalties for early termination
        if terminated:
            reward -= 1000.0
        if truncated:
            reward += 0.0  # No penalty for reaching max steps

        return float(reward)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step of the environment.

        Args:
            action: Action vector to apply.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Ensure action is numpy array with correct dtype
        action = np.array(action, dtype=np.float64)

        # Clip action to bounds
        action = np.clip(action, self.action_low, self.action_high)

        # Store previous action for info
        self.prev_action = action

        # Integrate system dynamics using Euler integration
        state_dot = self.A @ self.state + self.B @ action
        self.state = self.state + state_dot * self.dt

        # Increment time step
        self.current_step += 1

        # Cache reference for this step to avoid redundant calculations
        self._cached_reference = self._get_reference()

        # Get current observation
        observation = self._get_obs()

        # Check for termination due to bounds violation
        terminated = np.any(
            (observation < self.obs_low) | (observation > self.obs_high)
        )

        # Check for truncation due to max steps
        truncated = self.current_step >= self.max_steps

        # Compute reward
        reward = self._get_reward(self.state, action, terminated, truncated)

        # Get info dictionary (pass precomputed values to avoid redundant work)
        info = self._get_info_optimized(observation, reward)

        # Update termination flag for info
        self.terminated = terminated

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment (basic implementation).

        Args:
            mode: Rendering mode.

        Returns:
            None or rendered array depending on mode.
        """
        if mode == "human":
            print(
                f"Step: {self.current_step}, Time: {self.current_step * self.dt:.2f}s"
            )
            print(f"State: {self.state}")
            if self.prev_action is not None:
                print(f"Action: {self.prev_action}")
            print(f"Observation: {self._get_obs()}")
            if len(self.reference_config) > 0:
                ref = self._get_reference()
                print(f"Reference: {ref}")
            print("-" * 50)
        return None

    def close(self):
        """Clean up environment resources."""
        pass



