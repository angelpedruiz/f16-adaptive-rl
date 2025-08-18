import optuna
import yaml
from pathlib import Path
import numpy as np
import sys
from pathlib import Path

# Add the root project folder to sys.path (adjust if needed)
sys.path.append(str(Path(__file__).resolve().parent.parent))


from data.LinearF16SS import A_long_hi_ref as A, B_long_hi as B
from envs.f16_env import LinearModelF16
from agents.q_learning import QLearning
from utils.discretizer import UniformTileCoding
from utils import eval_utils  # all metric functions live here
from tqdm import tqdm
import pandas as pd



METRIC_FUNCTIONS = {
    "episodes_to_convergence": eval_utils.episodes_to_convergence,
    "episodes_to_convergence_settling_time": eval_utils.episodes_to_convergence_settling_time,
    "average_final_reward": eval_utils.average_final_reward,
    "success_rate": eval_utils.success_rate,
    "area_under_curve": eval_utils.area_under_curve,
    "reward_volatility": eval_utils.reward_volatility,
    "reward_slope": eval_utils.reward_slope,
    "stability_index": eval_utils.stability_index,
}



# ---------- CONFIG LOADING ----------
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------- TRAINING LOOP ----------
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
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated
            total_ep_reward += reward
        rewards.append(total_ep_reward)
        successes.append(not truncated)  # success = not truncated
    return np.array(rewards), np.array(successes)


# ---------- OBJECTIVE CREATOR ----------
def create_objective(config):
    target_metric = config["optimization"]["target_metric"]
    metric_args = config["optimization"]["metrics"].get(target_metric, {})

    def objective(trial):
        agent_config = config["agent"].copy()

        # Initialize obs_bins and action_bins as None (will set later)
        obs_bins_val = None
        action_bins_val = None

        for param_name, settings in config["optimization"]["params"].items():
            if not settings.get("enabled", True):
                continue

            if settings["type"] == "float":
                val = trial.suggest_float(param_name, settings["low"], settings["high"], log=settings.get("log", False))
            elif settings["type"] == "int":
                val = trial.suggest_int(param_name, settings["low"], settings["high"])
            else:
                raise ValueError(f"Unsupported param type: {settings['type']}")

            if param_name == "gamma":
                agent_config["discount_factor"] = val
            elif param_name == "epsilon_decay":
                agent_config["epsilon"]["decay"] = val
            elif param_name == "obs_bins":
                obs_bins_val = val
            elif param_name == "action_bins":
                action_bins_val = val
            else:
                agent_config[param_name] = val

        # If bins were optimized, replace full list with repeated values
        if obs_bins_val is not None:
            obs_dim = len(config["env"]["obs_low"])
            agent_config["obs_bins"] = [obs_bins_val] * obs_dim
        if action_bins_val is not None:
            agent_config["action_bins"] = [1, action_bins_val]

        # Create env and agent (same as before)
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
            learning_rate=agent_config["learning_rate"],
            initial_epsilon=agent_config["epsilon"]["start"],
            epsilon_decay=agent_config["epsilon"]["decay"],
            final_epsilon=agent_config["epsilon"]["final"],
            discount_factor=agent_config["discount_factor"],
            obs_discretizer=UniformTileCoding(env.observation_space, agent_config["obs_bins"]),
            action_discretizer=UniformTileCoding(env.action_space, agent_config["action_bins"]),
        )

        rewards, successes = run_training(
            agent,
            env,
            episodes=config["optimization"].get("episodes_per_trial", 50)
        )

        metric_fn = METRIC_FUNCTIONS[target_metric]

        if target_metric == "success_rate":
            return metric_fn(successes, **metric_args)
        else:
            return metric_fn(rewards, **metric_args)

    return objective


# ---------- MAIN ----------
if __name__ == "__main__":
    config_path = Path("config/q_learning.yaml")
    config = load_config(config_path)

    study = optuna.create_study(
        direction=config["optimization"].get("direction", "maximize"),
        sampler=optuna.samplers.TPESampler(seed=config["optimization"].get("seed", 42))
    )

    n_trials = config["optimization"].get("n_trials", 50)

    with tqdm(total=n_trials) as pbar:
        def progress_callback(study, trial):
            pbar.update(1)
            pbar.set_postfix_str(f"Best: {study.best_value:.4f} | Trial: {trial.number}")

        study.optimize(
            create_objective(config),
            n_trials=n_trials,
            callbacks=[progress_callback]
        )

    print("Best trial:")
    print(study.best_trial)

    print("Best hyperparameters:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")

    print(f"Best {config['optimization']['target_metric']}: {study.best_value}")

    # Save all trials to CSV
    trials_df = study.trials_dataframe()
    csv_path = Path("optuna_trials.csv")
    trials_df.to_csv(csv_path, index=False)
    print(f"All trial results saved to {csv_path}")


