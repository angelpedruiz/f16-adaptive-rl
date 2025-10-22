import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from tqdm import tqdm
import numpy as np
import yaml
from pathlib import Path
import time
import matplotlib.pyplot as plt

from agents.adhdp import ADHDPAgent
from data.LinearF16SS import A_long_hi_ref as A, B_long_hi as B, B_f1
from envs.f16_env import LinearModelF16
from utils.discretizer import UniformTileCoding
from utils.plots import plot_training_milestone_from_data
from utils.logging import setup_experiment_dir, save_run_summary
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.train_utils import save_checkpoint_if_needed, record_milestone_if_needed, save_milestone_plot, save_final_metrics
import torch

# -----------------------------
# Load config
# -----------------------------
with open("config/adhdp.yaml", "r") as f:
    config = yaml.safe_load(f)

agent_cfg = config["agent"]
env_cfg = config["env"]
training_cfg = config["training"]

# -----------------------------
# Create environment
# -----------------------------
# Create environment
env = LinearModelF16(
    A,
    B,
    max_steps=env_cfg["max_steps"],
    dt=env_cfg["dt"],
    reference_config=env_cfg["reference_config"],
    state_indices_for_obs=env_cfg["state_indices_to_keep"],
    action_low=env_cfg["action_low"],
    action_high=env_cfg["action_high"],
    obs_low=env_cfg["obs_low"],
    obs_high=env_cfg["obs_high"],
)
env = gym.wrappers.RecordEpisodeStatistics(
    env, buffer_length=config["training"]["episodes"]
)
# -----------------------------
# Instantiate ADHDP agent
# -----------------------------
agent = ADHDPAgent(
    obs_dim=agent_cfg["obs_dim"],
    act_dim=agent_cfg["act_dim"],
    action_low=env_cfg["action_low"],
    action_high=env_cfg["action_high"],
    hidden_dim=agent_cfg.get("hidden_dim", 32),
    num_layers=agent_cfg.get("num_layers", 2),
    actor_lr=agent_cfg.get("actor_lr", 1e-3),
    critic_lr=agent_cfg.get("critic_lr", 1e-3),
    gamma=agent_cfg.get("discount_factor", 0.99),
    noise=agent_cfg.get("noise", True),
    device="cpu"  # or "cuda" if GPU available
)

# -----------------------------
# Setup experiment folder / resume
# -----------------------------
exp_dir = setup_experiment_dir(config, algo_name="adhdp")
start_episode = 0
total_eps = config["training"]["episodes"]
checkpoint_interval = config["training"]["checkpoint_interval"]

# -----------------------------
# Training loop
# -----------------------------
# === Training Intro ===
print("🚀 Starting ADHDP Training")
print("=================================")
print(f"Agent:      ADHDP")
print(f"Episodes:   {total_eps}")
print(f"Seed:       {training_cfg['seed']}")
#print(f"Fault:      {config['online']['fault_type']}")
print("=================================\n")



start_time = time.time()
milestones = [min(int(frac * total_eps), total_eps - 1) for frac in config["training"]["milestones_fractions"]]

for episode in tqdm(range(start_episode, total_eps), desc="Training Progress", mininterval=5):
    obs, info = env.reset()
    done = False

    save_checkpoint_if_needed(agent, env, episode, checkpoint_interval, exp_dir, config)

    record_episode, state_trace, action_trace, reward_trace, reference_trace, error_trace = record_milestone_if_needed(episode, milestones, info)

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        cost = reward  # Convert reward to cost for ADHDP
        agent.update(obs, action, cost, terminated, next_obs)
        
        # Log terminated episode
        if terminated:
            violated_indices = np.where(
                (next_obs < env.env.obs_low) | (next_obs > env.env.obs_high)
            )[0]
            for idx in violated_indices:
                lower = env.env.obs_low[idx]
                upper = env.env.obs_high[idx]
                value = next_obs[idx]
                print(
                    f"Observation {idx} out of bounds at time {env.env.current_step * env.env.dt:.2f}s: "
                    f"value={value:.2f}, bounds=({lower:.2f}, {upper:.2f})"
                )

        if record_episode:
            state_trace.append(np.copy(info["state"]))
            action_trace.append(np.array(action, dtype=np.float64))
            reward_trace.append(reward)
            reference_trace.append(np.copy(info["reference"]))
            error_trace.append(np.copy(info["tracking_error"]))

        obs = next_obs
        done = terminated or truncated

    if record_episode:
        save_milestone_plot(exp_dir, episode, state_trace, action_trace, reference_trace, error_trace, reward_trace, tracked_indices=list(config["env"]["reference_config"].keys()))

# Save metrics and final checkpoint
save_final_metrics(agent, env, exp_dir, total_eps)

training_time = time.time() - start_time
print("\n✅ Training Complete!")
print(f"Total Episodes: {total_eps}")
print(f"Training Time:  {training_time/60:.1f} minutes")
print(f"Results saved in: {exp_dir}")
