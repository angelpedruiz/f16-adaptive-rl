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
from typing import Dict, Any, Optional

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


class Trainer:
    """
    Unified trainer class for F16 Adaptive RL project.

    Handles setup, configuration, and environment-specific training loops.
    Supports checkpointing, plotting, logging, and performance profiling.
    """

    def __init__(self, env_config_path: str, agent_config_path: str, resume_from: Optional[str] = None):
        """
        Initialize trainer with configuration files.

        Args:
            env_config_path: Path to environment configuration file
            agent_config_path: Path to agent configuration file
            resume_from: Optional checkpoint path to resume training
        """
        # Load and validate configuration
        self.env_config = load_config(env_config_path)
        self.agent_config = load_config(agent_config_path)
        self.resume_from = resume_from

        # Combine configs for validation
        self.config = {
            "agent": self.agent_config,
            "environment": self.env_config,
            "training": self.env_config.get("training", {}),
            "plotting": self.env_config.get("plotting", {}),
            "checkpointing": self.env_config.get("checkpointing", {}),
            "evaluation": self.env_config.get("evaluation", {})
        }
        validate_config(self.config)

        # Extract configuration sections
        self.training_config = self.config["training"]
        self.plotting_config = self.config["plotting"]
        self.checkpoint_config = self.config["checkpointing"]

        # Set random seed for reproducibility
        self._set_random_seed()

        # Create environment and agent
        self.env = self._create_environment()
        self.base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        self.agent = create_agent(self.agent_config, self.env)

        # Setup run directory
        self.run_dir = setup_run_directory(self.config, self.agent_config["type"], resume_from)

        # Initialize managers
        self.checkpoint_manager = CheckpointManager(
            self.run_dir, self.checkpoint_config, self.agent_config["type"]
        )
        self.plotting_manager = PlottingManager(self.run_dir, self.plotting_config)
        self.training_logger = TrainingLogger(self.run_dir, self.training_config)

        # Handle checkpoint resumption
        self.start_episode = 0
        if resume_from:
            self.start_episode = self.checkpoint_manager.load_checkpoint(
                resume_from, self.agent, self.env
            )
            self.training_logger.restore_from_checkpoint(self.checkpoint_manager)
            print(f"Resumed training from episode {self.start_episode}")
            print(f"Training metrics restored: {self.training_logger.is_resuming()}")

        # Training parameters
        self.total_episodes = self.training_config["episodes"]
        self.metrics_interval = self.plotting_config["training_metrics"].get("interval", 500)
        self.trajectories_interval = self.plotting_config["trajectories"].get("interval", 150)

        # Profiling data
        self.profiling = {
            "times_reset": [],
            "times_get_action": [],
            "times_env_step": [],
            "times_agent_update": [],
            "times_logging": [],
            "times_episode_total": []
        }

    def _set_random_seed(self) -> None:
        """Set random seeds for reproducibility."""
        if "seed" in self.training_config:
            np.random.seed(self.training_config["seed"])
            try:
                import torch
                torch.manual_seed(self.training_config["seed"])
            except ImportError:
                pass

    def _create_environment(self):
        """Create environment with RecordEpisodeStatistics wrapper."""
        env = create_environment(self.env_config)
        env = gym.wrappers.RecordEpisodeStatistics(
            env, buffer_length=self.training_config["episodes"]
        )
        return env

    def _print_training_header(self) -> None:
        """Print training session header."""
        if self.resume_from:
            print("Resuming F16 Adaptive RL Training")
            print("=" * 50)
            print(f"Agent:           {self.agent_config['type']}")
            print(f"Environment:     {self.env_config['name']}")
            print(f"Total Episodes:  {self.total_episodes}")
            print(f"Resume From:     Episode {self.start_episode}")
            print(f"Episodes Left:   {self.total_episodes - self.start_episode}")
            print(f"Resumed From:    {self.resume_from}")
            print(f"Results Dir:     {self.run_dir} (continuing)")
            print("=" * 50)
            print()
        else:
            print("Starting F16 Adaptive RL Training")
            print("=" * 50)
            print(f"Agent:           {self.agent_config['type']}")
            print(f"Environment:     {self.env_config['name']}")
            print(f"Total Episodes:  {self.total_episodes}")
            print(f"Start Episode:   {self.start_episode}")
            print(f"Results Dir:     {self.run_dir}")
            print("=" * 50)
            print()

    def _create_resumption_plot(self) -> None:
        """Create initial plot when resuming to show continuity."""
        if self.resume_from and self.start_episode > 0:
            try:
                self.plotting_manager.create_training_metrics_plot(
                    self.start_episode - 1, self.env, self.training_logger, is_final=False
                )
                print(f"Created resumption plot showing metrics up to episode {self.start_episode}")
            except Exception as e:
                print(f"Warning: Could not create resumption plot: {e}")

    def _should_record_trajectory(self, episode: int) -> bool:
        """Determine if trajectory should be recorded this episode."""
        return (
            (episode + 1) % self.trajectories_interval == 0
            and self.plotting_config.get("enabled", True)
            and self.plotting_config["trajectories"].get("save_data", True)
        )

    def _run_episode_step(self, obs, action, current_env, episode: int):
        """Execute a single environment step with profiling."""
        # Take step in environment
        step_start = time.time()
        next_obs, reward, terminated, truncated, info = current_env.step(action)
        if episode < 5:
            self.profiling["times_env_step"].append(time.time() - step_start)

        # Update agent
        update_start = time.time()
        self.agent.update(obs, action, reward, terminated, next_obs)
        if episode < 5:
            self.profiling["times_agent_update"].append(time.time() - update_start)

        return next_obs, reward, terminated, truncated, info

    def _post_episode_processing(self, episode: int, logging_start: float) -> None:
        """Handle post-episode logging, checkpointing, and plotting."""
        # Update training logger
        self.training_logger.log_episode(episode, self.env, self.agent)

        # Save checkpoint if needed
        if self.checkpoint_manager.should_save_checkpoint(episode):
            self.checkpoint_manager.save_checkpoint(episode, self.agent, self.env, self.training_logger)

        # Create training metrics plots
        if (episode + 1) % self.metrics_interval == 0 or episode == 0:
            self.plotting_manager.create_training_metrics_plot(episode, self.env, self.training_logger)

        # Decay exploration if agent supports it
        if hasattr(self.agent, "decay_epsilon"):
            self.agent.decay_epsilon()
        elif hasattr(self.agent, "decay_exploration"):
            self.agent.decay_exploration()

        # Track logging time
        if episode < 10:
            self.profiling["times_logging"].append(time.time() - logging_start)

    def _finalize_training(self, episode: int, start_time: float) -> None:
        """Save final results and print summary."""
        print("\nSaving final results...")

        # Final checkpoint
        final_episode = max(episode, self.start_episode)
        self.checkpoint_manager.save_checkpoint(
            final_episode, self.agent, self.env, self.training_logger, is_final=True
        )

        # Final plots
        self.plotting_manager.create_training_metrics_plot(
            final_episode, self.env, self.training_logger, is_final=True
        )

        # Training summary
        training_time = time.time() - start_time
        self.training_logger.save_final_summary(final_episode, training_time)

        # Create session summary
        create_session_summary(
            self.run_dir,
            self.config,
            final_episode,
            self.start_episode,
            training_time,
            self.resume_from,
            self.env,
            self.agent,
            self.training_logger,
        )

        # Performance analysis
        self._print_performance_analysis(training_time, final_episode)

    def _print_performance_analysis(self, training_time: float, final_episode: int) -> None:
        """Print detailed performance analysis."""
        if self.profiling["times_episode_total"]:
            avg_episode_time = np.mean(self.profiling["times_episode_total"])
            episodes_per_sec = 1.0 / avg_episode_time if avg_episode_time > 0 else 0

            print("\nPerformance Analysis:")
            print(f"Episodes completed: {len(self.profiling['times_episode_total'])}")
            print(f"Average episode time: {avg_episode_time * 1000:.1f}ms")
            print(f"Episodes per second: {episodes_per_sec:.2f}")
            print(f"Average reset time: {np.mean(self.profiling['times_reset']) * 1000:.2f}ms")
            print(f"Average action time: {np.mean(self.profiling['times_get_action']) * 1000:.2f}ms")
            print(f"Average env step time: {np.mean(self.profiling['times_env_step']) * 1000:.2f}ms")
            print(f"Average agent update time: {np.mean(self.profiling['times_agent_update']) * 1000:.2f}ms")
            print(f"Average logging time: {np.mean(self.profiling['times_logging']) * 1000:.2f}ms")

        print("\nTraining Complete!")
        if self.resume_from:
            print(f"Total Episodes Completed: {final_episode + 1}")
            print(f"Episodes in This Session: {final_episode + 1 - self.start_episode}")
            print(f"Session Training Time: {training_time / 60:.1f} minutes")
            print(f"Cumulative results saved in: {self.run_dir}")
        else:
            print(f"Total Episodes: {final_episode + 1}")
            print(f"Training Time:  {training_time / 60:.1f} minutes")
            print(f"Results saved in: {self.run_dir}")

    # =========================================================================
    # Environment-Specific Training Methods
    # =========================================================================

    def train_f16(self) -> None:
        """Training loop for F16 environment with full state tracking."""
        self._print_training_header()
        self._create_resumption_plot()

        start_time = time.time()
        episode = self.start_episode - 1

        try:
            for episode in tqdm(
                range(self.start_episode, self.total_episodes),
                desc="Training F16",
                mininterval=1.0,
            ):
                episode_start = time.time()

                # Reset environment
                reset_start = time.time()
                obs, info = self.env.reset()
                if episode < 10:
                    self.profiling["times_reset"].append(time.time() - reset_start)
                done = False

                # Trajectory recording setup
                record_data = self._should_record_trajectory(episode)
                if record_data:
                    state_trajectory = []
                    action_trajectory = []
                    reference_trajectory = []
                    errors_trajectory = []
                    reward_trajectory = []

                # Episode loop
                while not done:
                    # Get action
                    action_start = time.time()
                    action = self.agent.get_action(obs)
                    if episode < 5:
                        self.profiling["times_get_action"].append(time.time() - action_start)

                    # Step environment
                    next_obs, reward, terminated, truncated, info = self._run_episode_step(
                        obs, action, self.env, episode
                    )

                    # Record trajectory data
                    if record_data:
                        state_trajectory.append(
                            self.env.env.state.copy() if hasattr(self.env.env.state, "copy") else self.env.env.state
                        )
                        action_trajectory.append(action)
                        reference_trajectory.append(info.get("reference", None))
                        errors_trajectory.append(info.get("tracking_error", None))
                        reward_trajectory.append(reward)

                    obs = next_obs
                    done = terminated or truncated

                # Post-episode processing
                logging_start = time.time()
                self._post_episode_processing(episode, logging_start)

                # Create trajectory plot
                if record_data:
                    states = np.array(state_trajectory)
                    actions = np.array(action_trajectory)
                    save_path = self.plotting_manager.get_trajectory_path(episode)

                    self.plotting_manager.plot_f16_trajectory(
                        states=states,
                        actions=actions,
                        references=np.array(reference_trajectory),
                        errors=np.array(errors_trajectory),
                        rewards=reward_trajectory,
                        episode_num=episode,
                        save_path=str(save_path),
                        reference_config=getattr(self.base_env, "reference_config", None),
                    )

                if episode < 10:
                    self.profiling["times_episode_total"].append(time.time() - episode_start)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            self._finalize_training(episode, start_time)

    def train_lunarlander(self) -> None:
        """Training loop for LunarLander environment with video recording."""
        self._print_training_header()
        self._create_resumption_plot()

        # Create videos directory
        videos_dir = self.run_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        episode = self.start_episode - 1

        try:
            for episode in tqdm(
                range(self.start_episode, self.total_episodes),
                desc="Training LunarLander",
                mininterval=1.0,
            ):
                episode_start = time.time()

                # Setup video recording if needed
                record_data = self._should_record_trajectory(episode)
                if record_data:
                    video_env = gym.wrappers.RecordVideo(
                        self.env,
                        video_folder=str(videos_dir),
                        name_prefix=f"episode_{episode + 1}",
                        episode_trigger=lambda x: True
                    )
                    current_env = video_env
                    obs, info = current_env.reset()
                else:
                    current_env = self.env
                    reset_start = time.time()
                    obs, info = self.env.reset()
                    if episode < 10:
                        self.profiling["times_reset"].append(time.time() - reset_start)

                done = False

                # Episode loop
                while not done:
                    # Get action
                    action_start = time.time()
                    action = self.agent.get_action(obs)
                    if episode < 5:
                        self.profiling["times_get_action"].append(time.time() - action_start)

                    # Step environment
                    next_obs, reward, terminated, truncated, info = self._run_episode_step(
                        obs, action, current_env, episode
                    )

                    obs = next_obs
                    done = terminated or truncated

                # Close video recording if active
                if record_data:
                    current_env.close()

                # Post-episode processing
                logging_start = time.time()
                self._post_episode_processing(episode, logging_start)

                if episode < 10:
                    self.profiling["times_episode_total"].append(time.time() - episode_start)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            self._finalize_training(episode, start_time)

            # Video summary
            if self.plotting_config.get("enabled", True):
                video_files = list(videos_dir.glob("*.mp4"))
                if video_files:
                    print(f"Videos saved: {len(video_files)} episodes recorded in {videos_dir}")

    def train_shortperiod(self) -> None:
        """Training loop for short-period dynamics environment."""
        self._print_training_header()
        self._create_resumption_plot()

        start_time = time.time()
        episode = self.start_episode - 1

        try:
            for episode in tqdm(
                range(self.start_episode, self.total_episodes),
                desc="Training ShortPeriod",
                mininterval=1.0,
            ):
                episode_start = time.time()

                # Reset environment
                reset_start = time.time()
                obs, info = self.env.reset()
                if episode < 10:
                    self.profiling["times_reset"].append(time.time() - reset_start)
                done = False

                # Trajectory recording setup
                record_data = self._should_record_trajectory(episode)
                if record_data:
                    state_trajectory = []
                    action_trajectory = []
                    alpha_ref_trajectory = []
                    errors_trajectory = []
                    reward_trajectory = []

                # Episode loop
                while not done:
                    # Get action
                    action_start = time.time()
                    action = self.agent.get_action(obs)
                    if episode < 5:
                        self.profiling["times_get_action"].append(time.time() - action_start)

                    # Step environment
                    next_obs, reward, terminated, truncated, info = self._run_episode_step(
                        obs, action, self.env, episode
                    )

                    # Record trajectory data
                    if record_data:
                        state_trajectory.append(
                            self.env.env.state.copy() if hasattr(self.env.env.state, "copy") else self.env.env.state
                        )
                        action_trajectory.append(action)
                        alpha_ref_trajectory.append(info.get("alpha_ref", None))
                        errors_trajectory.append(info.get("tracking_error", None))
                        reward_trajectory.append(reward)

                    obs = next_obs
                    done = terminated or truncated

                # Post-episode processing
                logging_start = time.time()
                self._post_episode_processing(episode, logging_start)

                # Create trajectory plot
                if record_data and alpha_ref_trajectory:
                    states = np.array(state_trajectory)
                    actions = np.array(action_trajectory)
                    save_path = self.plotting_manager.get_trajectory_path(episode)

                    self.plotting_manager.plot_shortperiod_trajectory(
                        states=states,
                        actions=actions,
                        alpha_refs=np.array(alpha_ref_trajectory),
                        tracking_errors=np.array(errors_trajectory),
                        rewards=reward_trajectory,
                        episode_num=episode,
                        save_path=str(save_path),
                    )

                if episode < 10:
                    self.profiling["times_episode_total"].append(time.time() - episode_start)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            self._finalize_training(episode, start_time)

    def train(self) -> None:
        """
        Main training entry point that dispatches to environment-specific method.
        """
        env_name = self.env_config.get("name", "").lower()

        if "lunarlander" in env_name or "lunar" in env_name:
            self.train_lunarlander()
        elif "shortperiod" in env_name or "short" in env_name:
            self.train_shortperiod()
        elif "f16" in env_name:
            self.train_f16()
        else:
            # Default to F16 training for unknown environments
            print(f"Warning: Unknown environment '{env_name}', defaulting to F16 training loop")
            self.train_f16()


def train_agent(env_config_path: str, agent_config_path: str, resume_from: str = None):
    """
    Main training function that creates trainer and initiates training.

    Args:
        env_config_path: Path to the environment configuration file
        agent_config_path: Path to the agent configuration file
        resume_from: Optional path to checkpoint to resume training from
    """
    trainer = Trainer(env_config_path, agent_config_path, resume_from)
    trainer.train()


def main():
    """
    Main entry point for the training script.
    Handles command line arguments and initiates training.
    """
    parser = argparse.ArgumentParser(
        description="Train reinforcement learning agents on F16 control tasks"
    )

    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/environments/lunarlander.yaml",
        help="Path to the environment configuration file",
    )

    parser.add_argument(
        "--agent-config",
        type=str,
        default="configs/agents/idhp.yaml",
        help="Path to the agent configuration file",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    args = parser.parse_args()

    # Validate config files exist
    if not Path(args.env_config).exists():
        print(f"Environment configuration file not found: {args.env_config}")
        sys.exit(1)

    if not Path(args.agent_config).exists():
        print(f"Agent configuration file not found: {args.agent_config}")
        sys.exit(1)

    # Validate resume checkpoint if provided
    if args.resume_from and not Path(args.resume_from).exists():
        print(f"Resume checkpoint not found: {args.resume_from}")
        sys.exit(1)

    try:
        train_agent(args.env_config, args.agent_config, args.resume_from)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
