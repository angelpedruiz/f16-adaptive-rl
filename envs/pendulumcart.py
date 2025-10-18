import gymnasium as gym
import numpy as np
from typing import Optional

class PendulumCartEnv(gym.Env):
    """
    PendulumCartEnv
    ===============

    A Gymnasium environment simulating an **inverted pendulum on a cart**.  
    The goal is to balance the pendulum upright by applying a horizontal force to the cart.

    State:
        x = [x, x_dot, theta, theta_dot]
            x         – cart position [m]
            x_dot     – cart velocity [m/s]
            theta     – pendulum angle from upright [rad]
            theta_dot – angular velocity [rad/s]
            
    Observation: float32[4] → [x, x_dot, theta, theta_dot]

    Action:
        u ∈ ℝ → horizontal force on the cart [N]

    Dynamics (nonlinear):
        (M + m)ẍ + mlcosθθ̈ - mlsinθθ̇² = u  
        mlcosθẍ + ml²θ̈ - mglsinθ = 0

    Linearized around θ = 0 (upright):
        ẍ     = (1/M)u - (m*g/M)θ  
        θ̈     = -(1/(M*l))u + ((M+m)*g)/(M*l)θ

    Reward (default):
        r = - (θ² + 0.1θ̇² + 0.01x² + 0.001ẋ²)

    Termination:
        - |θ| > θ_max  
        - |x| > x_max  
        - episode length exceeded

    Spaces:
        Observation: Box(4,) → [x, x_dot, theta, theta_dot]  
        Action: Box(1,) → force u in [-u_max, u_max]
    """

    def __init__(self):
        super(PendulumCartEnv, self).__init__()

        # Physical parameters
        self.M = 1.0          # cart mass [kg]
        self.m = 0.1          # pendulum mass [kg]
        self.l = 0.5          # pendulum length [m]
        self.g = 10         # gravity [m/s²]

        # Control and simulation parameters
        self.dt = 0.02        # timestep [s]
        self.max_force = 10.0 # maximum force [N]

        # Limits
        self.x_max = 2.4      # max cart position [m]
        self.theta_max = 0.2  # max angle from upright [rad] (~11.5 degrees)

        # Linearized state-space model (around upright equilibrium)
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, -self.m * self.g / self.M, 0],
            [0, 0, 0, 1],
            [0, 0, (self.M + self.m) * self.g / (self.M * self.l), 0]
        ])

        self.B = np.array([
            [0],
            [1 / self.M],
            [0],
            [-1 / (self.M * self.l)]
        ])

        # Gymnasium spaces
        high = np.array([
            self.x_max * 2,   # x
            np.inf,           # x_dot
            np.pi,            # theta
            np.inf            # theta_dot
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-self.max_force,
            high=self.max_force,
            shape=(1,),
            dtype=np.float32
        )

        # State
        self.state = None
        self.timestep = 0


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Initialize cart at origin, pendulum near upright with small random perturbation
        x = 0.0
        x_dot = 0.0
        #theta = self.np_random.uniform(-0.1, 0.1)  # near upright
        theta = 0.1
        theta_dot = 0.0

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.timestep = 0

        observation = self.state.copy()
        info = {}

        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one timestep in the environment using linearized dynamics."""
        force = np.clip(action[0], -self.max_force, self.max_force)

        # Linearized dynamics: x_dot = A x + B u
        x_dot = self.A @ self.state + self.B.flatten() * force

        # Euler integration
        self.state = self.state + self.dt * x_dot
        self.state = self.state.astype(np.float32)
        self.timestep += 1

        x, x_dot_val, theta, theta_dot = self.state

        # Reward (minimize deviation from upright and center)
        reward = -(theta**2 + 0.1 * theta_dot**2 + 0.01 * x**2 + 0.001 * x_dot_val**2)

        terminated = bool(abs(theta) > self.theta_max or abs(x) > self.x_max)
        truncated = False

        return self.state.copy(), float(reward), terminated, truncated, {}



