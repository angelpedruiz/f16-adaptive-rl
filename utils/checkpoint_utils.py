import json
import os
import yaml
import numpy as np
from pathlib import Path


import json
import yaml
import numpy as np
from pathlib import Path

"""
Utility functions to save and load RL training checkpoints, including:
- Agent brain state (JSON)
- Episode metrics (NPY)
- Training config (YAML)
"""


def save_checkpoint(
    agent: object,
    current_episode: int,
    return_queue: list,
    length_queue: list,
    config: dict,
    checkpoint_dir: Path,
    filename: str = None,
):
    """
    Save checkpoint with agent brain (JSON), returns, lengths, training error (.npy),
    and configuration (YAML) inside a checkpoint folder.

    Args:
        agent (object): Agent instance with a get_brain() method
        current_episode (int): Current episode number
        return_queue (list): Episode returns from environment
        length_queue (list): Episode lengths from environment
        config (dict): Full training configuration to save for documentation
        checkpoint_dir (Path): Directory where checkpoint files will be saved
        filename (str, optional): JSON filename. Defaults to "checkpoint_ep{current_episode}.json"
    """

    # Create folder if it doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_ep{current_episode}.json"

    agent_brain = agent.get_brain()

    # Prepare brain dict for JSON serialization
    brain_serializable = {}
    for k, v in agent_brain.items():
        if isinstance(v, dict):
            brain_serializable[k] = {str(key): val for key, val in v.items()}
        elif isinstance(v, (np.ndarray, list)):
            brain_serializable[k] = np.array(v).tolist()
        else:
            brain_serializable[k] = v

    # Save agent brain JSON
    with open(checkpoint_dir / filename, "w") as f:
        json.dump(
            {
                "agent_brain": brain_serializable,
                "episode": current_episode,
            },
            f,
            indent=4,
        )
    print(f"Agent brain saved to {checkpoint_dir / filename}")

    # Save metrics as .npy
    np.save(checkpoint_dir / "returns.npy", np.array(return_queue))
    np.save(checkpoint_dir / "lengths.npy", np.array(length_queue))
    np.save(
        checkpoint_dir / "training_error.npy", np.array(agent_brain["training_error"])
    )
    print(f"Metrics saved to {checkpoint_dir} as .npy files")

    # Save training config
    with open(checkpoint_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Training config saved to {checkpoint_dir / 'config.yaml'}")


def load_checkpoint(checkpoint_file: Path):
    """
    Load agent brain and training state from a checkpoint JSON file,
    and load metrics from .npy files in the same folder.

    Args:
        checkpoint_file (Path): Path to the checkpoint JSON file

    Returns:
        dict: {
            "agent_brain": dict with keys (e.g. q_values, epsilon, etc.),
            "episode": int,
            "returns": np.ndarray,
            "lengths": np.ndarray,
            "training_error": np.ndarray
        }
    """
    # === Load JSON ===
    with open(checkpoint_file, "r") as f:
        checkpoint = json.load(f)

    raw_brain = checkpoint["agent_brain"]
    restored_brain = {}

    for k, v in raw_brain.items():
        if k == "q_values" and isinstance(v, dict):
            q_values_restored = {}
            for key_str, value in v.items():
                try:
                    key = eval(key_str)  # careful! only use if you trust the file
                except Exception:
                    key = key_str
                q_values_restored[key] = value
            restored_brain[k] = q_values_restored
        else:
            restored_brain[k] = v

    # === Load npy files from same directory ===
    checkpoint_dir = checkpoint_file.parent
    returns = np.load(checkpoint_dir / "returns.npy")
    lengths = np.load(checkpoint_dir / "lengths.npy")
    training_error = np.load(checkpoint_dir / "training_error.npy")

    return {
        "agent_brain": restored_brain,
        "episode": checkpoint["episode"],
        "returns": returns,
        "lengths": lengths,
        "training_error": training_error,
    }
