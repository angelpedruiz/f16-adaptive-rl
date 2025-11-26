"""
Minimal test to see the actor gradient chain rule verification output.
Run this to see the du/dparams verification at step 0.
"""
import sys
import numpy as np
import torch
from agents.IHDP.ihdp import IHDPAgent
from envs.shortperiod.shortperiod import ShortPeriodEnv

# Create environment and agent
env = ShortPeriodEnv()
agent = IHDPAgent(
    obs_space=env.obs_space,
    act_space=env.action_space,
    gamma=0.99,
    forgetting_factor=0.99,
    initial_covariance=1.0,
    hidden_sizes={'actor': [64, 64], 'critic': [64, 64]},
    learning_rates={'actor': 0.001, 'critic': 0.01},
    critic_weight_limit=30.0,
    actor_weight_limit=30.0,
)

# Run just a few steps to see the verification
obs, _ = env.reset()

for step in range(3):  # Only 3 steps to see output at step 0
    action = agent.get_action(obs)
    scaled_action = env.action_space.low + (0.5 * (action + 1.0) * (env.action_space.high - env.action_space.low)).astype(np.float32)
    next_obs, reward, terminated, truncated, info = env.step(scaled_action)

    print(f"\n{'='*80}")
    print(f"STEP {step} - Running update...")
    print(f"{'='*80}\n")

    metrics = agent.update(obs, action, reward, terminated, next_obs)

    if terminated or truncated:
        obs, _ = env.reset()
    else:
        obs = next_obs

print("\nDone! Check output above for ACTOR GRADIENT CHAIN RULE VERIFICATION section")
