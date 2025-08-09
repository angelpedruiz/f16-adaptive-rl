import sys
from pathlib import Path
import json  # <-- New import

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
import matplotlib.pyplot as plt
from utils.plots import reward_length_learning_error_plot
from utils.logging import load_metrics
from utils.eval_utils import compute_training_metrics

if __name__ == "__main__":
    # === Prompt for run ID ===
    run_id = input("Enter the path: ").strip()

    exp_dir = Path(run_id)

    if not exp_dir.exists():
        print(f"❌ Error: Directory '{exp_dir}' does not exist.")
        sys.exit(1)

    # === Load training metrics ===
    returns, lengths, training_error = load_metrics(exp_dir)

    # === Load config ===
    try:
        with open("config/q_learning.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Error: config/q_learning.yaml not found.")
        sys.exit(1)

    # === Extract evaluation parameters from config ===
    eval_cfg = config.get("eval", {})
    convergence_threshold = eval_cfg.get("convergence_threshold", 0.05)
    eval_window = eval_cfg.get("rolling_length", 50)
    last_n = eval_cfg.get("last_n", 50)
    settling_tolerance = eval_cfg.get("settling_tolerance", 0.05)
    settling_duration = eval_cfg.get("settling_duration", 100)

    # === Evaluate training metrics ===
    rewards = np.array(returns)
    terminations = np.ones_like(rewards, dtype=bool)  # Assuming all episodes successful
    training_metrics = compute_training_metrics(
        rewards=rewards,
        terminations=terminations,
        threshold=convergence_threshold,
        window=eval_window,
        last_n=last_n,
        settling_tolerance=settling_tolerance,
        settling_duration=settling_duration,
    )

    # === Save training metrics to JSON ===
    metrics_path = exp_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=4)
    print(f"✅ Saved training metrics to {metrics_path}")

    # === Dummy wrappers to reuse plotting function ===
    class DummyEnv:
        return_queue = returns
        length_queue = lengths

    class DummyAgent:
        training_error = training_error

    # === Plot the results ===
    rolling_length = eval_window
    plot_path = exp_dir / f"training_summary_rl{rolling_length}.png"
    reward_length_learning_error_plot(
        env=DummyEnv(),
        agent=DummyAgent(),
        rolling_length=rolling_length,
        save_path=plot_path,
    )
    print(f"✅ Saved training plot to {plot_path}")
