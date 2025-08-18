import optuna
import yaml
from pathlib import Path
import numpy as np
from data.LinearF16SS import A_long_hi_ref as A, B_long_hi as B
from envs.f16_env import LinearModelF16
from agents.q_learning import QLearning
from utils.discretizer import UniformTileCoding

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_objective(config):
    """
    Returns an Optuna objective function that builds agent+env and returns score.
    """
    def objective(trial):
        # Start from base agent config
        agent_config = config["agent"].copy()

        # Apply trial-suggested parameters
        for param_name, settings in config['optimization']['params'].items():
            if settings['type'] == 'float':
                val = trial.suggest_float(
                    param_name, settings['low'], settings['high'],
                    log=settings.get('log', False)
                )
            elif settings['type'] == 'int':
                val = trial.suggest_int(param_name, settings['low'], settings['high'])
            else:
                raise ValueError(f"Unsupported param type: {settings['type']}")

            # Map YAML param names to agent config keys if needed
            if param_name == "gamma":
                agent_config["discount_factor"] = val
            elif param_name == "epsilon_decay":
                agent_config["epsilon"]["decay"] = val
            else:
                agent_config[param_name] = val

        # Create env
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

        # Create agent with updated config
        agent = QLearning(
            env=env,
            learning_rate=agent_config["learning_rate"],
            initial_epsilon=agent_config["epsilon"]["start"],
            epsilon_decay=agent_config["epsilon"]["decay"],
            final_epsilon=agent_config["epsilon"]["final"],
            discount_factor=agent_config["discount_factor"],
            obs_discretizer=UniformTileCoding(
                env.observation_space, agent_config["obs_bins"]
            ),
            action_discretizer=UniformTileCoding(
                env.action_space, agent_config["action_bins"]
            ),
        )

        # Train & evaluate (shorter training for optimization)
        score = run_training(
            agent,
            env,
            episodes=config['optimization'].get('episodes_per_trial', 50)
        )

        return score

    return objective


def run_training(agent, env, episodes):
    rewards = []
    successes = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_ep_reward = 0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            done = terminated or truncated
            total_ep_reward += reward
        rewards.append(total_ep_reward)
        successes.append(not truncated)  # or however you define "success"
    return np.array(rewards), np.array(successes)
