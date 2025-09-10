import yaml
import numpy as np
from pathlib import Path
import ast  # ✅ Safe parsing

"""
Utility functions to save and load RL training checkpoints, including:
- Agent brain state (NPZ)
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
    Saves agent brain, metrics, and config to disk.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_ep{current_episode}"

    # === Save agent brain ===
    brain = agent.get_brain()
    # if the agent uses discretizers, save their params too
    if hasattr(agent, "obs_discretizer"):
        brain["obs_discretizer"] = agent.obs_discretizer.get_params()
        brain["action_discretizer"] = agent.action_discretizer.get_params()
    brain["episode"] = current_episode

    # Save as npz (auto-pickles complex Python objects like dict, lists, etc.)
    np.savez_compressed(checkpoint_dir / f"{filename}_brain.npz", **brain)
    print(f"\n🧠 Agent brain saved to {checkpoint_dir / f'{filename}_brain.npz'}")

    # === Save metrics ===
    np.save(checkpoint_dir / "returns.npy", np.array(return_queue))
    np.save(checkpoint_dir / "lengths.npy", np.array(length_queue))
    np.save(checkpoint_dir / "training_error.npy", np.array(brain["training_error"]))
    print(f"📈 Metrics saved to {checkpoint_dir}")

    # === Save config ===
    with open(checkpoint_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"⚙️  Training config saved to {checkpoint_dir / 'config.yaml'}")
    
def safe_eval_key(k):
    if isinstance(k, str):
        try:
            return ast.literal_eval(k)
        except (SyntaxError, ValueError):
            return k
    return k


def load_checkpoint(checkpoint_file: Path):
    """
    Loads agent brain, metrics, and config from disk.

    Args:
        checkpoint_file (Path): Path to the .npz file (e.g., "checkpoint_epXX_brain.npz")

    Returns:
        dict: Contains agent_brain, metrics, and discretizer configs
    """
    checkpoint_dir = checkpoint_file.parent

    # === Load agent brain ===
    npz_data = np.load(checkpoint_file, allow_pickle=True)
    agent_brain = {
        k: v.tolist() if isinstance(v, np.ndarray) and v.dtype == object else v
        for k, v in npz_data.items()
    }

    # Safely convert stringified Q-table keys back to tuples (if needed)
    if "q_values" in agent_brain and isinstance(agent_brain["q_values"], dict):
        restored_q = {}
        for k, v in agent_brain["q_values"].items():
            try:
                key = ast.literal_eval(k) if isinstance(k, str) else k
            except Exception:
                key = k  # fallback: keep as string
            restored_q[key] = v
        agent_brain["q_values"] = restored_q

    # === Load metrics ===
    returns = np.load(checkpoint_dir / "returns.npy")
    lengths = np.load(checkpoint_dir / "lengths.npy")
    training_error = np.load(checkpoint_dir / "training_error.npy")

    return {
        "agent_brain": agent_brain,
        "episode": agent_brain.get("episode", -1),
        "returns": returns,
        "lengths": lengths,
        "training_error": training_error,
        "obs_discretizer": agent_brain.get("obs_discretizer"),
        "action_discretizer": agent_brain.get("action_discretizer"),
    }
