import numpy as np


def episodes_to_convergence(
    rewards: np.ndarray, threshold: float = 0.05, window: int = 50
) -> int:
    """
    Calculate the number of episodes until the moving average of rewards stabilizes within a threshold.

    Args:
        rewards (np.ndarray): Array of episode-level rewards (i.e., total reward per episode).
        threshold (float, optional): Absolute difference threshold for convergence. Defaults to 0.05.
        window (int, optional): Number of episodes to average over. Defaults to 50.

    Returns:
        int: Index of the first episode where convergence occurs, or -1 if not found.
    """
    if len(rewards) < window:
        return -1

    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    for i in range(1, len(moving_avg)):
        if np.abs(moving_avg[i] - moving_avg[i - 1]) < threshold:
            return i + window - 1  # Adjust for the offset caused by convolution
    return -1


def episodes_to_convergence_settling_time(
    rewards: np.ndarray,
    tolerance_ratio: float = 0.05,
    stability_duration: int = 10,
    last_n: int = 50,
) -> int:
    """
    Calculate episodes to convergence using settling time approach:
    earliest episode after which rewards stay within tolerance band
    around steady-state average reward for `stability_duration` episodes.

    Args:
        rewards: Array of episode rewards.
        tolerance_ratio: Relative tolerance band (e.g., 0.05 = 5%).
        stability_duration: Number of consecutive episodes required inside band.
        last_n: Number of final episodes to compute steady-state reference.

    Returns:
        int: Episode index of convergence or -1 if not found.
    """
    if len(rewards) < stability_duration + last_n:
        return -1

    steady_state_ref = np.mean(rewards[-last_n:])
    lower_bound = steady_state_ref * (1 - tolerance_ratio)
    upper_bound = steady_state_ref * (1 + tolerance_ratio)

    # Check where rewards enter and stay in band for stability_duration episodes
    for start_idx in range(len(rewards) - stability_duration + 1):
        window = rewards[start_idx : start_idx + stability_duration]
        if np.all((window >= lower_bound) & (window <= upper_bound)):
            return start_idx  # Settling time found
    return -1


def average_final_reward(rewards: np.ndarray, last_n: int = 50) -> float:
    """
    Compute average reward over the last N episodes.

    Args:
        rewards (np.ndarray): Array of episode-level rewards.
        last_n (int, optional): Number of episodes to average. Defaults to 50.

    Returns:
        float: Mean reward over the last N episodes.
    """
    if len(rewards) < last_n:
        return float(np.mean(rewards))
    return float(np.mean(rewards[-last_n:]))


def success_rate(terminations: np.ndarray) -> float:
    """
    Calculate the percentage of episodes that terminated successfully (i.e., not early failure).

    Args:
        terminations (np.ndarray): Boolean array where True indicates successful (non-terminated) episode.

    Returns:
        float: Success rate (between 0 and 1).
    """
    return float(np.mean(terminations))


def area_under_curve(rewards: np.ndarray) -> float:
    """
    Computes area under the episode reward curve using the trapezoidal rule.

    Args:
        rewards (np.ndarray): Reward per episode.

    Returns:
        float: Area under the reward curve.
    """
    return np.trapz(rewards)



def reward_volatility(rewards: np.ndarray, window: int = 50) -> float:
    """
    Computes the standard deviation of residuals between rewards and their moving average (volatility).

    Args:
        rewards (np.ndarray): Reward per episode.
        window (int): Size of the moving average window.

    Returns:
        float: Standard deviation of the reward residuals.
    """
    if len(rewards) < window:
        # If not enough rewards for window, just return std of rewards
        return float(np.std(rewards))

    # Compute moving average (valid mode shortens the array)
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    # Align rewards to moving average length by trimming the start of rewards array
    aligned_rewards = rewards[window - 1:]

    # Compute residuals: difference between actual rewards and moving average
    residuals = aligned_rewards - moving_avg

    # Return std dev of residuals as volatility
    return float(np.std(residuals))



def reward_slope(rewards: np.ndarray) -> float:
    """
    Estimate the linear slope of the episode reward curve (improvement rate).

    Args:
        rewards (np.ndarray): Reward per episode.

    Returns:
        float: Slope of the linear regression fit.
    """
    if len(rewards) < 2:
        return 0.0
    x = np.arange(len(rewards))
    slope, _ = np.polyfit(x, rewards, 1)
    return float(slope)


def stability_index(rewards: np.ndarray, last_n: int = 50) -> float:
    """
    Computes 1 / std of last N episode rewards. Higher is more stable.

    Args:
        rewards (np.ndarray): Reward per episode.
        last_n (int): Number of episodes to consider.

    Returns:
        float: Stability index (higher = more stable).
    """
    if len(rewards) < 2:
        return 0.0
    segment = rewards[-last_n:] if len(rewards) >= last_n else rewards
    std = np.std(segment)
    return float(1.0 / (std + 1e-8))  # avoid division by zero


def compute_training_metrics(
    rewards: np.ndarray,
    terminations: np.ndarray = None,
    threshold: float = 0.05,
    window: int = 50,
    last_n: int = 50,
    settling_tolerance: float = 0.05,
    settling_duration: int = 100,
) -> dict:
    """
    Compute a collection of training metrics based on episode-level rewards and (optional) termination signals.

    Prints both convergence metrics: moving average difference and settling time.

    Returns:
        dict: Dictionary containing all computed training metrics.
    """
    conv_ep_ma_diff = episodes_to_convergence(
        rewards, threshold=threshold, window=window
    )
    conv_ep_settling = episodes_to_convergence_settling_time(
        rewards,
        tolerance_ratio=settling_tolerance,
        stability_duration=settling_duration,
        last_n=last_n,
    )

    metrics = {
        "average_final_reward": average_final_reward(rewards, last_n=last_n),
        "reward_slope": reward_slope(rewards),
        "reward_volatility": reward_volatility(rewards, window=window),
        "area_under_curve": area_under_curve(rewards),
        "episodes_to_convergence_moving_avg_diff": conv_ep_ma_diff,
        "episodes_to_convergence_settling_time": conv_ep_settling,
        "stability_index": stability_index(rewards, last_n=last_n),
    }

    if terminations is not None:
        metrics["success_rate"] = success_rate(terminations)

    print("\n=== Training Metrics Summary ===")
    print(f"Parameters:")
    print(f"  moving_avg_diff convergence threshold: {threshold}")
    print(f"  moving_avg_diff window: {window}")
    print(f"  settling_time tolerance: {settling_tolerance}")
    print(f"  settling_time duration: {settling_duration}")
    print(f"  last_n (for averages and stability): {last_n}\n")

    print(
        f"Reward stats: min={np.min(rewards):.2f}, max={np.max(rewards):.2f}, mean={np.mean(rewards):.2f}\n"
    )

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:32}: {v:.4g}")
        else:
            print(f"{k:32}: {v}")

    print("\nNotes:")
    print(
        "- 'episodes_to_convergence_moving_avg_diff' detects convergence by small changes in moving average."
    )
    print(
        "- 'episodes_to_convergence_settling_time' detects convergence as stability inside a tolerance band for consecutive episodes."
    )
    print("- 'reward_slope' shows average reward improvement per episode.")
    print("- 'reward_volatility' measures variability in rewards from local moving average.")
    print(
        "- 'stability_index' = 1/std of last N rewards; higher means more stable training."
    )
    print(
        "- 'success_rate' assumes all episodes ended successfully (modify if needed)."
    )
    print("- Metrics scale depends on your reward magnitude.\n")

    return metrics
