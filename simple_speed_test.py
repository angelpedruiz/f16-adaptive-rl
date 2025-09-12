#!/usr/bin/env python3
"""
Simple speed test - directly tests the core training loop components.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

def test_basic_loop():
    """Test just the core environment and agent interaction."""
    print("Testing basic training loop...")
    
    # Import components
    from utils.environment_factory import create_environment
    from utils.agent_factory import create_agent
    
    # Simple config
    env_config = {
        "name": "f16",
        "max_steps": 100,
        "dt": 0.01,
        "state_indices_for_obs": [4],
        "action_low": [0, -22.5],
        "action_high": [0, 27.0],
        "obs_low": [-2.0, -50.0],
        "obs_high": [2.0, 50.0],
        "reference_config": {
            1: {
                "type": "cos_step",
                "amplitude": {"min": -20.0, "max": 20.0, "n_levels": 5},
                "T_step": 1.0,
                "step_duration": {"min": 1.0, "max": 1.0, "n_levels": 1},
                "transition_duration": {"min": 0.5, "max": 0.5, "n_levels": 1}
            }
        }
    }
    
    agent_config = {
        "type": "q_learning",
        "learning_rate": 0.4,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.1,
        "discount_factor": 0.95,
        "obs_discretizer": {
            "type": "uniform_tile_coding",
            "bins": [5, 5]
        },
        "action_discretizer": {
            "type": "uniform_tile_coding",
            "bins": [1, 5]
        }
    }
    
    # Create components
    print("Creating environment and agent...")
    env = create_environment(env_config)
    agent = create_agent(agent_config, env)
    
    # Test episodes
    num_episodes = 5
    episode_times = []
    
    print(f"Running {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        episode_start = time.time()
        
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated
            step_count += 1
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        print(f"Episode {episode + 1}: {episode_time*1000:.1f}ms ({step_count} steps)")
    
    # Results
    avg_episode_time = np.mean(episode_times)
    episodes_per_sec = 1.0 / avg_episode_time if avg_episode_time > 0 else 0
    
    print(f"\nResults:")
    print(f"Average episode time: {avg_episode_time*1000:.1f}ms")
    print(f"Episodes per second: {episodes_per_sec:.2f}")
    print(f"Total test time: {sum(episode_times):.2f}s")

def main():
    """Run the simple performance test."""
    try:
        test_basic_loop()
        print("\nSimple speed test completed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()