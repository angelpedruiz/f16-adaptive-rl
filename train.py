"""
Unified training script for f16-adaptive-rl project.
Supports all agents, environments, and fully config-driven training with checkpointing, plotting, and statistics.
"""

import sys
import os
import time
import yaml
import json
import numpy as np
import gymnasium as gym
from pathlib import Path
from tqdm import tqdm
import importlib
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from utils.config_utils import load_config, validate_config
from utils.environment_factory import create_environment
from utils.agent_factory import create_agent
from utils.checkpoint_manager import CheckpointManager
from utils.plotting_manager import PlottingManager
from utils.training_logger import TrainingLogger
from utils.session_summary import create_session_summary


def setup_run_directory(config: dict, agent_name: str, resume_from: str = None) -> Path:
    """
    Create unique run directory for this training session or return existing one if resuming.

    Args:
        config: Training configuration
        agent_name: Name of the agent being trained
        resume_from: Optional path to checkpoint to resume from

    Returns:
        Path to the run directory
    """
    if resume_from:
        # Extract run directory from checkpoint path
        checkpoint_path = Path(resume_from)
        if checkpoint_path.is_file():
            run_dir = (
                checkpoint_path.parent.parent
            )  # checkpoint file -> checkpoint dir -> run dir
        else:
            run_dir = checkpoint_path.parent  # checkpoint dir -> run dir

        # Update config for this resumed run (allow changing parameters like total episodes)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return run_dir
    else:
        # Create new run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

        run_dir = Path("experiments") / agent_name / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config for this run
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return run_dir


def train_agent(config_path: str, resume_from: str = None):
    """
    Main training function that orchestrates the entire training process.

    Args:
        config_path: Path to the YAML configuration file
        resume_from: Optional path to checkpoint to resume training from
    """
    # Load and validate configuration
    config = load_config(config_path)
    validate_config(config)

    # Extract configuration sections
    agent_config = config["agent"]
    env_config = config["environment"]
    training_config = config["training"]
    plotting_config = config["plotting"]
    checkpoint_config = config["checkpointing"]

    # Set random seed for reproducibility
    if "seed" in training_config:
        np.random.seed(training_config["seed"])
        # Set PyTorch seed if available
        try:
            import torch

            torch.manual_seed(training_config["seed"])
        except ImportError:
            pass

    # Create environment with RecordEpisodeStatistics wrapper
    env = create_environment(env_config)
    env = gym.wrappers.RecordEpisodeStatistics(
        env, buffer_length=training_config["episodes"]
    )

    # For performance testing - access the unwrapped environment
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    # Create agent
    agent = create_agent(agent_config, env)

    # Setup run directory
    run_dir = setup_run_directory(config, agent_config["type"], resume_from)

    # Initialize managers
    checkpoint_manager = CheckpointManager(
        run_dir, checkpoint_config, agent_config["type"]
    )

    plotting_manager = PlottingManager(run_dir, plotting_config)

    training_logger = TrainingLogger(run_dir, training_config)

    # Create videos directory for lunar lander environments
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if environment is lunar lander type
    is_lunar_lander = "lunar" in env_config.get("name", "").lower() or "lunarlander" in env_config.get("name", "").lower()

    # Handle resume from checkpoint
    start_episode = 0
    if resume_from:
        start_episode = checkpoint_manager.load_checkpoint(resume_from, agent, env)
        # Restore training logger state from checkpoint
        training_logger.restore_from_checkpoint(checkpoint_manager)
        print(f"Resumed training from episode {start_episode}")
        print(f"Training metrics restored: {training_logger.is_resuming()}")

    # Training parameters
    total_episodes = training_config["episodes"]

    # Plotting configuration
    metrics_interval = plotting_config["training_metrics"].get("interval", 500)
    trajectories_interval = plotting_config["trajectories"].get("interval", 150)

    # Print training summary
    if resume_from:
        print("Resuming F16 Adaptive RL Training")
        print("=" * 50)
        print(f"Agent:           {agent_config['type']}")
        print(f"Environment:     {env_config['name']}")
        print(f"Total Episodes:  {total_episodes}")
        print(f"Resume From:     Episode {start_episode}")
        print(f"Episodes Left:   {total_episodes - start_episode}")
        print(f"Resumed From:    {resume_from}")
        print(f"Results Dir:     {run_dir} (continuing)")
        print("=" * 50)
        print()
    else:
        print("Starting F16 Adaptive RL Training")
        print("=" * 50)
        print(f"Agent:           {agent_config['type']}")
        print(f"Environment:     {env_config['name']}")
        print(f"Total Episodes:  {total_episodes}")
        print(f"Start Episode:   {start_episode}")
        print(f"Results Dir:     {run_dir}")
        print("=" * 50)
        print()

    # Training loop with profiling
    start_time = time.time()

    # Profiling variables
    times_reset = []
    times_get_action = []
    times_env_step = []
    times_agent_update = []
    times_logging = []
    times_episode_total = []

    # Initialize episode variable to prevent UnboundLocalError
    episode = start_episode - 1

    # Create initial plots if resuming to show continuity
    if resume_from and start_episode > 0:
        try:
            plotting_manager.create_training_metrics_plot(
                start_episode - 1, env, training_logger, is_final=False
            )
            print(
                f"Created resumption plot showing metrics up to episode {start_episode}"
            )
        except Exception as e:
            print(f"Warning: Could not create resumption plot: {e}")

    try:
        for episode in tqdm(
            range(start_episode, total_episodes),
            desc="Training Progress",
            mininterval=1.0,
        ):
            episode_start = time.time()

            # Reset environment for new episode
            reset_start = time.time()
            obs, info = env.reset()
            if episode < 10:  # Only profile first 10 episodes
                times_reset.append(time.time() - reset_start)
            done = False

            # Record trajectories for non-lunar lander or videos for lunar lander
            record_data = (
                (episode + 1) % trajectories_interval == 0
                and plotting_config.get("enabled", True)
                and plotting_config["trajectories"].get("save_data", True)
            )

            if record_data and not is_lunar_lander:
                # Pre-allocate arrays for trajectory recording (non-lunar lander)
                max_episode_steps = training_config.get("max_steps", 3000)
                state_trajectory = []
                action_trajectory = []
                reference_trajectory = []
                errors_trajectory = []
                reward_trajectory = []
                step_count = 0
            elif record_data and is_lunar_lander:
                # Setup video recording for lunar lander
                video_env = gym.wrappers.RecordVideo(
                    env,
                    video_folder=str(videos_dir),
                    name_prefix=f"episode_{episode + 1}",
                    episode_trigger=lambda x: True
                )
                current_env = video_env
                obs, info = current_env.reset()
            else:
                current_env = env

            # Episode loop
            while not done:
                # Get action from agent
                action_start = time.time()
                action = agent.get_action(obs)
                # Only collect profiling data for first few episodes to prevent memory issues
                if episode < 5:
                    times_get_action.append(time.time() - action_start)

                # Take step in environment (use current_env for video recording)
                step_start = time.time()
                next_obs, reward, terminated, truncated, info = current_env.step(action)
                if episode < 5:
                    times_env_step.append(time.time() - step_start)

                # Update agent
                update_start = time.time()
                agent.update(obs, action, reward, terminated, next_obs)
                if episode < 5:
                    times_agent_update.append(time.time() - update_start)

                # Record trajectories only for non-lunar lander environments
                if record_data and not is_lunar_lander:
                    # Use copy to avoid reference issues and optimize access
                    state_trajectory.append(env.env.state.copy() if hasattr(env.env.state, "copy") else env.env.state)
                    action_trajectory.append(action)
                    reference_trajectory.append(info.get("reference", None))
                    errors_trajectory.append(info.get("tracking_error", None))
                    reward_trajectory.append(reward)
                    step_count += 1
                    
                # Move to next state
                obs = next_obs
                done = terminated or truncated

            # Close video recording if it was active
            if record_data and is_lunar_lander:
                current_env.close()

            # Post-episode processing with timing
            logging_start = time.time()

            # Update training logger
            training_logger.log_episode(episode, env, agent)

            # Save checkpoint if needed
            if checkpoint_manager.should_save_checkpoint(episode):
                checkpoint_manager.save_checkpoint(episode, agent, env, training_logger)

            # Create training metrics plots
            if (episode + 1) % metrics_interval == 0 or episode == 0:
                plotting_manager.create_training_metrics_plot(
                    episode, env, training_logger
                )

            # Create trajectory plots (only for non-lunar lander when data was recorded)
            if record_data and not is_lunar_lander:
                states = np.array(state_trajectory)
                actions = np.array(action_trajectory)
                save_path = plotting_manager.get_trajectory_path(episode)
                plotting_manager.plot_test_episode_trajectory(
                    states=states,
                    actions=actions,
                    references=np.array(reference_trajectory),
                    errors=np.array(errors_trajectory),
                    rewards=reward_trajectory,
                    episode_num=episode,
                    save_path=str(save_path),
                    reference_config=getattr(base_env, "reference_config", None),
                )

            # Decay exploration parameters if agent supports it
            if hasattr(agent, "decay_epsilon"):
                agent.decay_epsilon()
            elif hasattr(agent, "decay_exploration"):
                agent.decay_exploration()

            if episode < 10:  # Only profile first 10 episodes
                times_logging.append(time.time() - logging_start)
                times_episode_total.append(time.time() - episode_start)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        # Always save final checkpoint and plots
        print("\nSaving final results...")

        # Final checkpoint (ensure episode is valid)
        final_episode = max(episode, start_episode)
        checkpoint_manager.save_checkpoint(
            final_episode, agent, env, training_logger, is_final=True
        )

        # Final plots
        plotting_manager.create_training_metrics_plot(
            final_episode, env, training_logger, is_final=True
        )

        # Training summary
        training_time = time.time() - start_time
        training_logger.save_final_summary(final_episode, training_time)

        # Create session summary
        create_session_summary(
            run_dir,
            config,
            final_episode,
            start_episode,
            training_time,
            resume_from,
            env,
            agent,
            training_logger,
        )

        # Performance analysis
        if times_episode_total and len(times_episode_total) > 0:
            avg_episode_time = np.mean(times_episode_total)
            episodes_per_sec = 1.0 / avg_episode_time if avg_episode_time > 0 else 0

            print("\nPerformance Analysis:")
            print(f"Episodes completed: {len(times_episode_total)}")
            print(f"Average episode time: {avg_episode_time * 1000:.1f}ms")
            print(f"Episodes per second: {episodes_per_sec:.2f}")
            print(f"Average reset time: {np.mean(times_reset) * 1000:.2f}ms")
            print(f"Average action time: {np.mean(times_get_action) * 1000:.2f}ms")
            print(f"Average env step time: {np.mean(times_env_step) * 1000:.2f}ms")
            print(
                f"Average agent update time: {np.mean(times_agent_update) * 1000:.2f}ms"
            )
            print(f"Average logging time: {np.mean(times_logging) * 1000:.2f}ms")

        print("\nTraining Complete!")
        if resume_from:
            print(f"Total Episodes Completed: {max(episode + 1, start_episode)}")
            print(f"Episodes in This Session: {max(episode + 1 - start_episode, 0)}")
            print(f"Session Training Time: {training_time / 60:.1f} minutes")
            print(f"Cumulative results saved in: {run_dir}")
        else:
            print(f"Total Episodes: {max(episode + 1, start_episode)}")
            print(f"Training Time:  {training_time / 60:.1f} minutes")
            print(f"Results saved in: {run_dir}")

        # Video summary for lunar lander
        if is_lunar_lander and plotting_config.get("enabled", True):
            video_files = list(videos_dir.glob("*.mp4"))
            if video_files:
                print(f"Videos saved: {len(video_files)} episodes recorded in {videos_dir}")


def main():
    """
    Main entry point for the training script.
    Handles command line arguments and initiates training.
    """
    parser = argparse.ArgumentParser(
        description="Train reinforcement learning agents on F16 control tasks"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/q_learning_config.yaml",
        help="Path to the YAML configuration file (default: config/q_learning.yaml)",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    args = parser.parse_args()

    # Validate config file exists
    if not Path(args.config).exists():
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)

    # Validate resume checkpoint if provided
    if args.resume_from and not Path(args.resume_from).exists():
        print(f"Resume checkpoint not found: {args.resume_from}")
        sys.exit(1)

    try:
        train_agent(args.config, args.resume_from)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
