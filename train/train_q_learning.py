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
from data.LinearF16SS import A_long_hi_ref as A, B_long_hi as B, B_f1
from envs.f16_env import LinearModelF16
from utils.discretizer import UniformTileCoding
from utils.plots import plot_training_milestone_from_data
from utils.logging import setup_experiment_dir, save_run_summary
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.train_utils import save_checkpoint_if_needed, record_milestone_if_needed, save_milestone_plot, save_final_metrics

with open("config/q_learning.yaml", "r") as f:
    config = yaml.safe_load(f)
    
np.random.seed(config["training"]["seed"])

# Create environment
env = LinearModelF16(
    A,
    B,
    max_steps=config["env"]["max_steps"],
    dt=config["env"]["dt"],
    reference_config=config["env"]["reference_config"],
    state_indices_for_obs=config["env"]["state_indices_to_keep"],
    action_low=config["env"]["action_low"],
    action_high=config["env"]["action_high"],
    obs_low=config["env"]["obs_low"],
    obs_high=config["env"]["obs_high"],
)
env = gym.wrappers.RecordEpisodeStatistics(
    env, buffer_length=config["training"]["episodes"]
)

agent = QLearning(
    env=env,
    learning_rate=config["agent"]["learning_rate"],
    initial_epsilon=config["agent"]["epsilon"]["start"],
    epsilon_decay=config["agent"]["epsilon"]["decay"],
    final_epsilon=config["agent"]["epsilon"]["final"],
    discount_factor=config["agent"]["discount_factor"],
    obs_discretizer=UniformTileCoding(env.observation_space, config["agent"]["obs_bins"]),
    action_discretizer=UniformTileCoding(env.action_space, config["agent"]["action_bins"]),
)

# Setup experiment folder / resume
resume_from = config["training"].get("resume_from", None)
if resume_from:
    ckpt_path = Path(resume_from).resolve()
    exp_dir = ckpt_path.parent.parent
    ckpt = load_checkpoint(ckpt_path)
    agent.load_brain(ckpt["agent_brain"])
    agent.obs_discretizer.set_params(ckpt["obs_discretizer"])
    agent.action_discretizer.set_params(ckpt["action_discretizer"])
    agent.training_error = ckpt["training_error"].tolist()
    env.return_queue = ckpt["returns"].tolist()
    env.length_queue = ckpt["lengths"].tolist()
    start_episode = ckpt["episode"] + 1
    print(f"🔄 Resuming training from checkpoint: {resume_from}")
    print(f"   Episodes completed: {start_episode}/{config['training']['episodes']}")
    total_eps = config["training"]["episodes"] + start_episode
else:
    exp_dir = setup_experiment_dir(config, algo_name="q_learning")
    start_episode = 0
    total_eps = config["training"]["episodes"]

checkpoint_interval = config["training"]["checkpoint_interval"]


# === Training Intro ===
print("🚀 Starting Q-Learning Training")
print("=================================")
print(f"Agent:      QLearning")
print(f"Episodes:   {total_eps}")
print(f"Seed:       {config['training']['seed']}")
print(f"Fault:      {config['online']['fault_type']}")
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
