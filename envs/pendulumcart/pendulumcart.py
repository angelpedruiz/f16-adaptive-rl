import gymnasium as gym
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.signal import place_poles

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

    def __init__(self, dt: float = 0.01, max_steps: int = 1000):
        super().__init__()

        # Physical parameters
        self.M = 1.0       # cart mass [kg]
        self.m = 0.1          # pendulum mass [kg]
        self.l = 0.5*2          # pendulum length [m]
        self.g = 10         # gravity [m/s²]

        # Control and simulation parameters
        self.dt = dt      # timestep [s]
        self.max_steps = max_steps
        
        # Limits
        self.x_max = 10      # max cart position [m]
        self.theta_max = np.pi/2  # max angle from upright [rad] (~11.5 degrees)
        self.max_force = 10.0 # maximum force [N]

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
        
        self.state_dim = 4
        self.act_dim = 1

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
        info = {'state': self.state.copy()}

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
        reward = -np.abs(theta) + 0.1

        terminated = bool(abs(theta) > self.theta_max or abs(x) > self.x_max)
        if self.timestep >= self.max_steps:
            truncated = True
        else:
            truncated = False
        
        info = {'state': self.state.copy(), 'action': force, 'reward': reward}

        return self.state.copy(), float(reward), terminated, truncated, info
    
    def get_params(self) -> dict:
        """Return environment parameters."""
        params = {
            'M': self.M,
            'm': self.m,
            'l': self.l,
            'g': self.g,
            'dt': self.dt,
            'max_steps': self.max_steps,
            'max_force': self.max_force,
            'x_max': self.x_max,
            'theta_max': self.theta_max
        }
        return params

    def render(self):
        """Render the current state of the pendulum cart."""
        if not hasattr(self, 'fig'):
            # Initialize figure on first render
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            plt.ion()  # Interactive mode

        self.ax.clear()

        # Extract state
        x, _, theta, _ = self.state

        # Set axis limits
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-1, 1.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        # Draw track
        self.ax.plot([-3, 3], [-0.5, -0.5], 'k-', linewidth=2)

        # Draw cart (centered at x position)
        cart_width = 0.3
        cart_height = 0.2
        cart = Rectangle((x - cart_width/2, -0.5), cart_width, cart_height,
                        facecolor='blue', edgecolor='black', linewidth=2)
        self.ax.add_patch(cart)

        # Draw pendulum rod
        pend_x = x + self.l * np.sin(theta)
        pend_y = -0.5 + cart_height + self.l * np.cos(theta)
        self.ax.plot([x, pend_x], [-0.5 + cart_height, pend_y],
                    'r-', linewidth=3)

        # Draw pendulum bob
        bob = Circle((pend_x, pend_y), 0.08,
                    facecolor='red', edgecolor='black', linewidth=2)
        self.ax.add_patch(bob)

        # Add state information
        info_text = f'x = {x:.3f} m\nθ = {theta:.3f} rad ({np.degrees(theta):.1f}°)\nStep = {self.timestep}'
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Add title
        self.ax.set_title('Inverted Pendulum on Cart', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Position (m)')
        self.ax.set_ylabel('Height (m)')

        plt.pause(0.001)
        plt.draw()

    def close(self):
        """Close the rendering window."""
        if hasattr(self, 'fig'):
            plt.close(self.fig)