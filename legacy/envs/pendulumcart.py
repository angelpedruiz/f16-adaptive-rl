import gymnasium as gym
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.signal import place_poles
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.hdp import HDPAgent

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
        reward = -(theta**2)

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

def run_episode(env: gym.Env, agent, max_steps: int = 200) -> float:
    """Run a single episode using the given agent in the environment."""
    obs, info = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)

        obs = next_obs
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward

# Simple episode with rendering
if __name__ == "__main__":
    print("=" * 60)
    print("Running Episode with Detailed Logging (HDP)")
    print("=" * 60)

    # Create environment
    env = PendulumCartEnv()

    # Initialize ADHDP agent
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set deterministic seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    agent = HDPAgent(
        obs_dim=env.state_dim,
        act_dim=env.act_dim,
        hidden_sizes=[32, 32],
        actor_lr=1e-2,      # Learning rate for actor (was 1e-4, too slow)
        critic_lr=5e-3,     # Higher learning rate for critic
        model_lr=1e-2,      # Highest for model (supervised learning)
        gamma=0.95,         # Standard discount factor
        device=device,
        action_low=np.array([-env.max_force]),
        action_high=np.array([env.max_force])
    )

    # Reset environment
    obs, info = env.reset(seed=42)
    print("\nInitial State:")
    print(f"  x         = {obs[0]:.6f} m")
    print(f"  x_dot     = {obs[1]:.6f} m/s")
    print(f"  theta     = {obs[2]:.6f} rad ({np.degrees(obs[2]):.3f}°)")
    print(f"  theta_dot = {obs[3]:.6f} rad/s")

    # Render initial state
    env.render()

    # Run episode with detailed logging for 10 steps
    max_steps = 200
    total_reward = 0

    print(f"\n{'='*80}")
    print(f"RUNNING {max_steps} STEPS WITH DETAILED LOGGING")
    print(f"{'='*80}\n")

    for step in range(max_steps):
        print(f"\n{'='*80}")
        print(f"STEP {step + 1}/{max_steps}")
        print(f"{'='*80}")

        # Print full state vector before taking action
        print(f"\n--- STATE BEFORE ACTION ---")
        print(f"State vector: [{obs[0]:.6f}, {obs[1]:.6f}, {obs[2]:.6f}, {obs[3]:.6f}]")
        print(f"  x         = {obs[0]:.6f} m")
        print(f"  x_dot     = {obs[1]:.6f} m/s")
        print(f"  theta     = {obs[2]:.6f} rad ({np.degrees(obs[2]):.3f}°)")
        print(f"  theta_dot = {obs[3]:.6f} rad/s")

        # Get action from HDP agent with detailed logging
        print(f"\n--- ACTION SELECTION ---")
        action = agent.get_action(obs)

        # Also show the raw actor output for debugging
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_normalized = agent.actor(obs_tensor)
            action_normalized_np = action_normalized.cpu().numpy().flatten()

        print(f"Actor output (tanh, [-1,1]): {action_normalized_np}")
        print(f"Scaled action (force):        {action[0]:.6f} N")

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print transition tuple
        print(f"\n--- TRANSITION TUPLE ---")
        print(f"state:      [{obs[0]:.6f}, {obs[1]:.6f}, {obs[2]:.6f}, {obs[3]:.6f}]")
        print(f"action:     [{action[0]:.6f}]")
        print(f"reward:     {reward:.6f}")
        print(f"next_state: [{next_obs[0]:.6f}, {next_obs[1]:.6f}, {next_obs[2]:.6f}, {next_obs[3]:.6f}]")
        print(f"done:       {terminated}")

        # Update agent - use the agent's update method directly
        print(f"\n{'-'*80}")
        print(f"CALLING AGENT UPDATE METHOD")
        print(f"{'-'*80}")

        # Convert to tensors for logging
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

        print(f"\n--- PREDICTIONS BEFORE UPDATE ---")
        with torch.no_grad():
            # Current V-value
            if agent.prev_obs is not None:
                V_prev = agent.critic(agent.prev_obs)
                print(f"Critic V(s_prev):             {V_prev.item():.6f}")
            else:
                print(f"Critic V(s_prev):             None (first step)")

            V_current = agent.critic(obs_t)
            print(f"Critic V(s):                  {V_current.item():.6f}")

            # Model prediction
            model_input = torch.cat([obs_t, action_t], dim=-1)
            next_state_pred = agent.model(model_input)
            print(f"Model predicted next state:   {next_state_pred.cpu().numpy().flatten()}")
            print(f"Actual next state:            {next_obs}")

        # Call agent update
        agent.update(obs, action, reward, terminated, next_obs)

        print(f"\n--- PREDICTIONS AFTER UPDATE ---")
        with torch.no_grad():
            # New predictions
            new_V_current = agent.critic(obs_t)
            new_actor_output = agent.actor(obs_t)

            # New model prediction
            new_model_input = torch.cat([obs_t, action_t], dim=-1)
            new_next_state_pred = agent.model(new_model_input)

            print(f"New Critic V(s):              {new_V_current.item():.6f}")
            print(f"New Actor output:             {new_actor_output.cpu().numpy().flatten()}")
            print(f"New Model next state pred:    {new_next_state_pred.cpu().numpy().flatten()}")
            print(f"Model prediction error:       {np.linalg.norm(new_next_state_pred.cpu().numpy().flatten() - next_obs):.6f}")

        print(f"\n{'-'*80}")
        print(f"END OF UPDATE")
        print(f"{'-'*80}")

        # Update observation
        obs = next_obs

        # Render the current state
        env.render()

        # Check if episode ended
        if terminated or truncated:
            print(f"\n{'!'*80}")
            print(f"EPISODE ENDED AT STEP {step + 1}")
            print(f"{'!'*80}")
            if terminated:
                print("Reason: TERMINATED")
                print(f"  |theta| = {abs(obs[2]):.6f} rad (limit: {env.theta_max:.6f} rad)")
                print(f"  |x|     = {abs(obs[0]):.6f} m   (limit: {env.x_max:.6f} m)")
            if truncated:
                print("Reason: TRUNCATED")
            break

    print(f"\n{'='*80}")
    print(f"EPISODE COMPLETE")
    print(f"{'='*80}")
    print(f"Total steps:  {step + 1}")
    print(f"Total reward: {total_reward:.6f}")
    print(f"{'='*80}\n")

    print("\nClose the plot window to exit...")
    # Keep window open
    plt.ioff()
    plt.show()

    env.close()

