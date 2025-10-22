import yaml
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.q_learning import QLearning
from envs.f16_env import LinearModelF16
from utils.test_utils import plot_test_episode
from data.LinearF16SS import A_long_hi_ref as A, B_long_hi as B
from utils.checkpoint_utils import load_checkpoint
from utils.discretizer import UniformTileCoding


# === 1. Load config ===
with open("config/q_learning.yaml", "r") as f:
    config = yaml.safe_load(f)

ckpt_path = Path(config["test"]["checkpoint_path"])
assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"


# === 4. Init environment ===
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


# === 5. Load agent with checkpointed brain ===
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

# === 2. Load checkpoint ===

ckpt = load_checkpoint(ckpt_path)
agent.load_brain(ckpt["agent_brain"])
agent.obs_discretizer.set_params(ckpt["obs_discretizer"])
agent.action_discretizer.set_params(ckpt["action_discretizer"])
agent.epsilon = 0.0

# === 6. Run test episode ===
states, actions, refs, errors, rewards = [], [], [], [], []

obs, info = env.reset()
done = False

while not done:
    action = agent.get_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)

    # Store trajectory
    states.append(np.copy(info["state"]))
    actions.append(np.array(action, dtype=np.float64))
    refs.append(np.copy(info["reference"]))
    errors.append(np.copy(info["tracking_error"]))
    rewards.append(reward)

    obs = next_obs
    done = terminated or truncated

states = np.array(states)
actions = np.array(actions)
refs = np.array(refs)
errors = np.array(errors)
rewards = np.array(rewards)

# === 5. Plot milestone ===
plot_test_episode(
    states,
    actions,
    refs,
    errors,
    rewards,
    episode_id="eval",
    save_path=ckpt_path.parent / "test_episode_plot.png",
)


print(f"Saved plot to {ckpt_path.parent / 'test_episode_plot.png'}")
