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

    def __init__(self):
        super(PendulumCartEnv, self).__init__()

        # Physical parameters
        self.M = 1.0          # cart mass [kg]
        self.m = 0.1          # pendulum mass [kg]
        self.l = 0.5          # pendulum length [m]
        self.g = 10         # gravity [m/s²]

        # Control and simulation parameters
        self.dt = 0.02        # timestep [s]
        self.max_force = 100.0 # maximum force [N]

        # Limits
        self.x_max = 10      # max cart position [m]
        self.theta_max = np.pi/2  # max angle from upright [rad] (~11.5 degrees)

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
        theta = 0.5
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

def linear_feedback_control(A: np.ndarray, B: np.ndarray, eigs: np.ndarray, state: np.ndarray) -> float:
    """Compute control force using LQR."""
    # Desired poles for closed-loop system
    K = place_poles(A, B, eigs).gain_matrix
    force = -K @ state
    return force[0]

# Simple episode with rendering
if __name__ == "__main__":
    print("=" * 60)
    print("Running Episode with Rendering")
    print("=" * 60)

    # Create environment
    env = PendulumCartEnv()

    # Reset environment
    obs, info = env.reset(seed=42)
    print("\nInitial State:")
    print(f"  x     = {obs[0]:.4f} m")
    print(f"  theta = {obs[2]:.4f} rad ({np.degrees(obs[2]):.2f}°)")

    # Render initial state
    env.render()

    # Run episode
    max_steps = 100
    total_reward = 0

    print(f"\nRunning episode for max {max_steps} steps...")
    print("Close the plot window to end early.\n")

    for step in range(max_steps):
        # get action
        force = linear_feedback_control(env.A, env.B, eigs=[-2, -2.5, -3, -3.5], state=obs)
        
        # clip action
        max_force = env.max_force
        force = np.clip(force, -max_force, max_force)
        action = np.array([force])

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render
        env.render()

        # Log every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}: x={obs[0]:.3f}, theta={obs[2]:.3f} rad, reward={reward:.3f}")

        # Check if episode ended
        if terminated or truncated:
            print(f"\nEpisode ended at step {step + 1}")
            if terminated:
                print("Reason: TERMINATED")
                print(f"  |theta| = {abs(obs[2]):.4f} rad (limit: {env.theta_max:.4f} rad)")
                print(f"  |x|     = {abs(obs[0]):.4f} m   (limit: {env.x_max:.4f} m)")
                if abs(obs[2]) > env.theta_max:
                    print(f"  -> Pendulum angle exceeded limit!")
                if abs(obs[0]) > env.x_max:
                    print(f"  -> Cart position exceeded limit!")
            if truncated:
                print("Reason: TRUNCATED (max steps reached)")
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    print("\nClose the window to exit...")

    # Keep window open
    plt.ioff()
    plt.show()

    env.close()

