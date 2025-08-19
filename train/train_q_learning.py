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

from agents.q_learning import QLearning
from data.LinearF16SS import A_long_hi_ref as A, B_long_hi as B, A_f1
from envs.f16_env import LinearModelF16
from utils.discretizer import UniformTileCoding
from utils.plots import plot_training_milestone_from_data
from utils.logging import setup_experiment_dir, save_run_summary
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.train_utils import setup_training, apply_fault
from utils.train_utils import save_checkpoint_if_needed, record_milestone_if_needed, save_milestone_plot, save_final_metrics

with open("config/q_learning.yaml", "r") as f:
    config = yaml.safe_load(f)

env, agent, exp_dir, start_episode, total_eps, fault_type = setup_training(config)
checkpoint_interval = config["training"]["checkpoint_interval"]


# === Training Intro ===
print("🚀 Starting Q-Learning Training")
print("=================================")
print(f"Agent:      QLearning")
print(f"Episodes:   {total_eps}")
print(f"Seed:       {config['training']['seed']}")
print(f"Fault:      {fault_type}")
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
        next_obs, reward, terminated, truncated, info = env.env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)
        
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

    agent.decay_epsilon()

    if record_episode:
        save_milestone_plot(exp_dir, episode, state_trace, action_trace, reference_trace, error_trace, reward_trace, tracked_indices=list(config["env"]["reference_config"].keys()))

# Save metrics and final checkpoint
save_final_metrics(agent, env, exp_dir, total_eps)

training_time = time.time() - start_time
print("\n✅ Training Complete!")
print(f"Total Episodes: {total_eps}")
print(f"Training Time:  {training_time/60:.1f} minutes")
print(f"Results saved in: {exp_dir}")
