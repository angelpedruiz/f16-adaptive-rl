from pathlib import Path
import yaml
import json
import datetime
import numpy as np
import time


def setup_experiment_dir(config, algo_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{algo_name}/run_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    return exp_dir


def save_run_summary(
    exp_dir,
    config,
    agent,
    env,
    training_time_sec=None,
    start_episode=None,
    total_eps=None,
):
    summary_path = exp_dir / "summary.json"

    rewards = np.array(env.return_queue)
    lengths = np.array(env.length_queue)

    sprint_summary = {
        "sprint_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "episodes": config["training"]["episodes"],
        "start_episode": start_episode,
        "end_episode": (total_eps - 1) if total_eps is not None else None,
        "final_epsilon_value": agent.epsilon,
        "avg_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "reward_std": float(np.std(rewards)),
        "avg_episode_length": float(np.mean(lengths)),
    }

    if training_time_sec is not None:
        sprint_summary["training_time_sec"] = round(training_time_sec, 2)

    # Agent hyperparameters
    agent_params = {}
    for attr in dir(agent):
        if not attr.startswith("_") and not callable(getattr(agent, attr)):
            value = getattr(agent, attr)
            if isinstance(value, (int, float, str, bool)):
                agent_params[attr] = value

    sprint_summary["agent_hyperparameters"] = agent_params

    # Load existing summaries (if any)
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary_list = json.load(f)
    else:
        summary_list = []

    summary_list.append(sprint_summary)

    with open(summary_path, "w") as f:
        json.dump(summary_list, f, indent=2)


def load_metrics(exp_dir):
    returns = np.load(exp_dir / "returns.npy")
    lengths = np.load(exp_dir / "lengths.npy")
    training_error = np.load(exp_dir / "training_error.npy")
    return returns, lengths, training_error
