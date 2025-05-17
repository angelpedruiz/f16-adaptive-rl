import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional

class InvertedPendulumEnv(gym.Env):
    def __init__(self, max_steps: int = 500):
        # Pendulum parameters (simplified)
        self.mass = 1.0  # mass of the pendulum
        self.length = 1.0  # length of the pendulum
        self.inertia = (1/3) * self.mass * self.length ** 2
        self.g = 9.81  # acceleration due to gravity
        self.torque_max = 2  # max torque the agent can apply
        self.dt = 0.01  # time step for simulation
        self.max_steps = max_steps  # max steps per episode
        self.lower_state_bounds = np.array([-np.pi/4, -2.0])
        self.upper_state_bounds = np.array([np.pi/4, 2.0])

        self.action_space = spaces.Box(low=-self.torque_max, high=self.torque_max, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=self.lower_state_bounds, 
                                            high=self.upper_state_bounds, 
                                            dtype=np.float64)

        self.state = np.array([0.1, 0.0])  # [angle, angular_velocity]
        self.step_count = 0

    def _get_obs(self):
        # Return the current state as observation
        return tuple(self.state)
    
    def _get_info(self):
        # Return additional information (if any)
        return {}
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        margin_ratio = 0.25  # 20% margin inside the bounds
        low = self.lower_state_bounds
        high = self.upper_state_bounds

        margin = (high - low) * margin_ratio
        safe_low = low + margin
        safe_high = high - margin

        self.state = np.random.uniform(safe_low, safe_high)
        self.step_count = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action) -> tuple:
        # Apply the action (torque)
        torque = action[0]

        # Simulate the pendulum dynamics
        angle, angular_velocity = self.state

        # Pendulum physics (simplified 2nd order dynamics for an inverted pendulum)
        theta_dot = angular_velocity
        theta_ddot = ((self.length * self.mass * self.g) * np.sin(angle) / 2 + torque) / self.inertia

        # Update the state using simple Euler integration
        angle += theta_dot * self.dt
        angular_velocity += theta_ddot * self.dt

        # Update the state
        self.state = np.array([angle, angular_velocity])

        # Increment the step count
        self.step_count += 1

        # Termination condition: if the angle exceeds a certain threshold (i.e., the pendulum falls)
        terminated = bool(np.abs(angle) > np.pi / 2)

        # Truncation condition: if the max steps are reached
        truncated = self.step_count >= self.max_steps

        # Reward: penalize large angles and high angular velocities
        reward = -np.abs(angle)

        # If the episode is terminated or truncated, return the appropriate info
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
gym.register(
    id="gymnasium_env/InvertedPendulum-v0",
    entry_point=InvertedPendulumEnv,
)