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
from data.LinearF16SS import A_long_hi_ref as A, B_long_hi as B
from envs.f16_env import LinearModelF16
from utils.discretizer import UniformTileCoding
from utils.plots import plot_training_milestone_from_data
from utils.logging import setup_experiment_dir, save_run_summary
from utils.checkpoint_utils import save_checkpoint, load_checkpoint


# === Load config ===
with open("config/q_learning.yaml", "r") as f:
    config = yaml.safe_load(f)

checkpoint_interval = config["training"].get("checkpoint_interval", 1000)



# === Set random seed for reproducibility ===
np.random.seed(config["training"]["seed"])


# === Create Environment ===
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

# === Create Agent ===
agent = QLearning(
    env=env,
    learning_rate=config["agent"]["learning_rate"],
    initial_epsilon=config["agent"]["epsilon"]["start"],
    epsilon_decay=config["agent"]["epsilon"]["decay"],
    final_epsilon=config["agent"]["epsilon"]["final"],
    discount_factor=config["agent"]["discount_factor"],
    obs_discretizer=UniformTileCoding(
        env.observation_space, config["agent"]["obs_bins"]
    ),
    action_discretizer=UniformTileCoding(
        env.action_space, config["agent"]["action_bins"]
    ),
)

start_time = time.time()

# === Setup experiment folder ===
RESUME_FROM = config["training"].get("resume_from", None)




if RESUME_FROM:
    ckpt_path = Path(RESUME_FROM).resolve()
    exp_dir = ckpt_path.parent.parent  # checkpoint_epXXX → run_xxxxxx
    ckpt = load_checkpoint(ckpt_path)
    agent.load_brain(ckpt["agent_brain"])
    agent.obs_discretizer.set_params(ckpt["obs_discretizer"])
    agent.action_discretizer.set_params(ckpt["action_discretizer"])
    env.return_queue = ckpt["returns"].tolist()
    env.length_queue = ckpt["lengths"].tolist()
    start_episode = ckpt["episode"] + 1
    print(f"Resuming training from {RESUME_FROM} in existing directory {exp_dir}")
    total_eps = config["training"]["episodes"] + start_episode
else:
    exp_dir = setup_experiment_dir(config, algo_name="q_learning")
    start_episode = 0
    total_eps = config["training"]["episodes"]



# === Training Loop ===
milestones = [
    min(int(frac * total_eps), total_eps - 1)
    for frac in config["training"]["milestones_fractions"]
]


for episode in tqdm(range(start_episode, total_eps)):
    obs, info = env.reset()
    done = False

    # Save a checkpoint every N episodes
    if episode != 0 and episode % checkpoint_interval == 0:
        checkpoint_dir = exp_dir / f"checkpoint_ep{episode}"
        save_checkpoint(
            agent=agent,
            current_episode=episode,
            return_queue=env.return_queue,
            length_queue=env.length_queue,
            config=config,
            checkpoint_dir=checkpoint_dir,
        )

    # Record Snapshot Epsiode Evolution
    record_episode = episode in milestones
    if record_episode:
        state_trace = []
        action_trace = []
        reward_trace = []
        state_trace.append(np.copy(info["state"]))
        reference_trace = [np.copy(info["reference"])]
        error_trace = [np.copy(info["tracking_error"])]

    # === Run episode ===
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)

        if record_episode:
            state_trace.append(np.copy(info["state"]))
            action_trace.append(np.array(action, dtype=np.float64))
            reward_trace.append(reward)

            reference_trace.append(np.copy(info["reference"]))
            error_trace.append(np.copy(info["tracking_error"]))

        obs = next_obs
        done = terminated or truncated

    agent.decay_epsilon()
    # =========================

    if record_episode:
        milestone_path = exp_dir / f"state_action_evolution_ep{episode + 1}.png"
        plot_training_milestone_from_data(
            np.array(state_trace),
            np.array(action_trace),
            np.array(reference_trace),
            np.array(error_trace),
            reward_trace,
            episode_num=episode,
            save_path=milestone_path,
            tracked_indices=list(config["env"]["reference_config"].keys()),
        )


# === Save metrics for post-training analysis ===
np.save(exp_dir / "returns.npy", np.array(env.return_queue))
np.save(exp_dir / "lengths.npy", np.array(env.length_queue))
np.save(exp_dir / "training_error.npy", np.array(agent.training_error))

# === Final Checkpoint ===
final_ckpt_dir = exp_dir / f"checkpoint_final_ep{total_eps - 1}"
save_checkpoint(
    agent=agent,
    current_episode=total_eps - 1,
    return_queue=env.return_queue,
    length_queue=env.length_queue,
    config=config,
    checkpoint_dir=final_ckpt_dir,
)


# === Final Report ===
training_time = time.time() - start_time
save_run_summary(exp_dir, config, agent, env, training_time_sec=training_time, start_episode=start_episode, total_eps=total_eps)

print(f"\nTraining complete. Results saved to: {exp_dir}")
