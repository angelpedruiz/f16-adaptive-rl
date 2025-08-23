import yaml
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.LinearF16SS import A_long_hi_ref as A, B_long_hi as B
from agents.q_learning import QLearning
from envs.f16_env import LinearModelF16
from utils.checkpoint_utils import load_checkpoint
from utils.discretizer import UniformTileCoding
from utils.test_utils import plot_test_episode
from utils.online_utils import apply_fault
'''
The purpose of this script is to simulate and analyse online learning starting from a offline trained checkpoint on the nominal case.
1. Load config and checkpoint
1. Initialise Environemnt and Agent
2. Run online episode on fault (learning or freezed) (may be more than one parallel runs to account for variations)
3. Output some type of analysis (state action evolution, statistics)
'''

# 1. Load Config and Checkpoint

# === 1. Load config ===
with open("config/q_learning.yaml", "r") as f:
    config = yaml.safe_load(f)

ckpt_path = Path(config["online"]["checkpoint_path"])
assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

# 2. Init Env and Agent
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

ckpt = load_checkpoint(ckpt_path)
agent.load_brain(ckpt["agent_brain"])
agent.obs_discretizer.set_params(ckpt["obs_discretizer"])
agent.action_discretizer.set_params(ckpt["action_discretizer"])

# Run Episode

states, actions, refs, errors, rewards = [], [], [], [], []
obs, info = env.reset(options={'fault_type': config['online']['fault_type']})
done = False
print(f'B matrix: {env.B}')

while not done:
    action = agent.get_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    if config['online']['learn']:
        agent.update(obs, action, reward, terminated, next_obs)

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
    states, actions, refs, errors, rewards,
    episode_id="eval",
    save_path=ckpt_path.parent / "online_episode_plot.png"
)


print(f"Saved plot to {ckpt_path.parent / 'online_episode_plot.png'}")