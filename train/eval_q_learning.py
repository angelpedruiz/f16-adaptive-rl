import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
import matplotlib.pyplot as plt
from utils.plots import reward_length_learning_error_plot
from utils.logging import load_metrics

if __name__ == "__main__":
    # === Prompt for run ID ===
    run_id = input("Enter the run ID (e.g., 20250805_101230): ").strip()
    exp_dir = Path(f"experiments/q_learning/run_{run_id}")

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

    # === Prompt for rolling window (optional) ===
    window_input = input("Enter rolling average window (or press Enter to use default): ").strip()
    if window_input.isdigit():
        rolling_length = int(window_input)
    else:
        rolling_length = config.get("training", {}).get("rolling_length", 50)

    # === Dummy wrappers to reuse plotting function ===
    class DummyEnv:
        return_queue = returns
        length_queue = lengths

    class DummyAgent:
        training_error = training_error

    # === Plot the results ===
    reward_length_learning_error_plot(
        env=DummyEnv(),
        agent=DummyAgent(),
        rolling_length=rolling_length,
        save_path=exp_dir / f"training_summary_rl{rolling_length}.png"
    )
