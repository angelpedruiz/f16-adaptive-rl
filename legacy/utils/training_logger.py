"""
Training logger for tracking and saving training statistics and metrics.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import gymnasium as gym


class TrainingLogger:
    """
    Logs training progress and statistics throughout training.
    
    Tracks:
    - Episode-wise statistics (rewards, lengths, training errors)
    - Agent-specific metrics (epsilon, learning rates, etc.)
    - Training time and performance metrics
    - Custom metrics defined by agents
    """
    
    def __init__(self, run_dir: Path, training_config: Dict[str, Any]):
        """
        Initialize training logger.
        
        Args:
            run_dir: Directory for this training run
            training_config: Training configuration
        """
        self.run_dir = Path(run_dir)
        self.training_config = training_config
        
        # Store references to env and agent for direct access
        self.env = None
        self.agent = None
        
        # Initialize storage for custom metrics only
        self.custom_metrics = {}
        
        # Agent-specific metrics
        self.epsilon_values = []
        self.learning_rates = []
        
        # Timing information
        self.episode_times = []
        self.start_time = datetime.now()
        
        # Configuration (optimized default for better performance)
        self.save_frequency = training_config.get("log_frequency", 500)
        self.detailed_logging = training_config.get("detailed_logging", True)
        
        # Track if we're resuming from checkpoint
        self.is_resumed = False
        
        # Load existing logs if they exist (for resuming)
        self._load_existing_logs()
    
    def log_episode(self, episode: int, env: gym.Env, agent: object) -> None:
        """
        Log statistics for a completed episode.
        
        Args:
            episode: Episode number
            env: Environment (with RecordEpisodeStatistics wrapper)
            agent: Agent that completed the episode
        """
        # Store references for direct access
        self.env = env
        self.agent = agent
        
        # Minimize logging overhead - only log metrics when really needed
        log_metrics_now = (
            episode % 50 == 0 or 
            episode == 0 or 
            (episode + 1) % self.save_frequency == 0
        )
        
        if log_metrics_now:
            # Log agent-specific metrics (optimized)
            if hasattr(agent, 'epsilon'):
                self.epsilon_values.append(float(agent.epsilon))
            
            # Skip custom metrics unless explicitly enabled
            if self.detailed_logging and hasattr(agent, 'get_metrics'):
                metrics = agent.get_metrics()
                for key, value in metrics.items():
                    if key not in self.custom_metrics:
                        self.custom_metrics[key] = []
                    self.custom_metrics[key].append(value)
        
        # Save logs periodically
        if (episode + 1) % self.save_frequency == 0:
            self.save_logs()
    
    def restore_from_checkpoint(self, checkpoint_manager) -> None:
        """
        Restore training logger state from checkpoint data.
        
        Args:
            checkpoint_manager: CheckpointManager with restored statistics
        """
        if hasattr(checkpoint_manager, 'get_restored_stats'):
            restored_stats = checkpoint_manager.get_restored_stats()
            
            # Restore custom metrics and agent metrics
            if 'training_errors' in restored_stats:
                # Don't store in custom_metrics as it's handled by agent
                pass
                
            if 'epsilon_values' in restored_stats:
                self.epsilon_values = list(restored_stats['epsilon_values'])
                print(f"Restored {len(self.epsilon_values)} epsilon values")
                
            if 'learning_rates' in restored_stats:
                self.learning_rates = list(restored_stats['learning_rates'])
                print(f"Restored {len(self.learning_rates)} learning rate values")
            
            # Mark as resumed
            self.is_resumed = True
    
    def log_training_error(self, error: float) -> None:
        """
        Log training error (e.g., TD error, loss).
        
        Note: This method is kept for backwards compatibility.
        Training errors are now accessed directly from agent.training_error.
        
        Args:
            error: Training error value
        """
        pass
    
    def log_custom_metric(self, name: str, value: Any) -> None:
        """
        Log a custom metric.
        
        Args:
            name: Name of the metric
            value: Value of the metric
        """
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        self.custom_metrics[name].append(value)
    
    def get_training_stats(self) -> Dict[str, List]:
        """
        Get all tracked training statistics.
        
        Returns:
            Dictionary containing all tracked metrics
        """
        stats = {
            'epsilon_values': self.epsilon_values,
            'learning_rates': self.learning_rates,
        }
        
        # Get episode data from environment queues
        if self.env and hasattr(self.env, 'return_queue'):
            stats['episode_rewards'] = list(self.env.return_queue)
        
        if self.env and hasattr(self.env, 'length_queue'):
            stats['episode_lengths'] = list(self.env.length_queue)
        
        # Get training errors from agent
        if self.agent and hasattr(self.agent, 'training_error'):
            if isinstance(self.agent.training_error, list):
                stats['training_errors'] = self.agent.training_error
            else:
                stats['training_errors'] = [self.agent.training_error]
        
        # Add custom metrics
        stats.update(self.custom_metrics)
        
        return stats
    
    def is_resuming(self) -> bool:
        """Check if this logger is resuming from a checkpoint."""
        return self.is_resumed
    
    def get_recent_performance(self, n_episodes: int = 100) -> Dict[str, float]:
        """
        Get performance metrics for recent episodes.
        
        Args:
            n_episodes: Number of recent episodes to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        performance = {}
        
        # Get episode rewards from environment
        if self.env and hasattr(self.env, 'return_queue') and len(self.env.return_queue) > 0:
            recent_rewards = list(self.env.return_queue)[-n_episodes:]
            performance['mean_reward'] = float(np.mean(recent_rewards))
            performance['std_reward'] = float(np.std(recent_rewards))
            performance['max_reward'] = float(np.max(recent_rewards))
            performance['min_reward'] = float(np.min(recent_rewards))
        
        # Get episode lengths from environment
        if self.env and hasattr(self.env, 'length_queue') and len(self.env.length_queue) > 0:
            recent_lengths = list(self.env.length_queue)[-n_episodes:]
            performance['mean_length'] = float(np.mean(recent_lengths))
            performance['std_length'] = float(np.std(recent_lengths))
        
        # Get training errors from agent
        if self.agent and hasattr(self.agent, 'training_error'):
            if isinstance(self.agent.training_error, list) and len(self.agent.training_error) > 0:
                recent_errors = self.agent.training_error[-n_episodes:]
                performance['mean_training_error'] = float(np.mean(recent_errors))
                performance['std_training_error'] = float(np.std(recent_errors))
            elif not isinstance(self.agent.training_error, list):
                performance['mean_training_error'] = float(self.agent.training_error)
                performance['std_training_error'] = 0.0
        
        return performance
    
    def save_logs(self) -> None:
        """Save current logs to files."""
        # Save episode-wise data from environment queues
        if self.env and hasattr(self.env, 'return_queue') and len(self.env.return_queue) > 0:
            np.save(self.run_dir / "episode_rewards.npy", np.array(list(self.env.return_queue)))
        
        if self.env and hasattr(self.env, 'length_queue') and len(self.env.length_queue) > 0:
            np.save(self.run_dir / "episode_lengths.npy", np.array(list(self.env.length_queue)))
        
        # Save training errors from agent
        if self.agent and hasattr(self.agent, 'training_error'):
            if isinstance(self.agent.training_error, list) and len(self.agent.training_error) > 0:
                np.save(self.run_dir / "training_errors.npy", np.array(self.agent.training_error))
            elif not isinstance(self.agent.training_error, list):
                np.save(self.run_dir / "training_errors.npy", np.array([self.agent.training_error]))
        
        # Save agent metrics
        if self.epsilon_values:
            np.save(self.run_dir / "epsilon_values.npy", np.array(self.epsilon_values))
        
        if self.learning_rates:
            np.save(self.run_dir / "learning_rates.npy", np.array(self.learning_rates))
        
        # Save custom metrics
        for name, values in self.custom_metrics.items():
            if values:
                try:
                    np.save(self.run_dir / f"{name}.npy", np.array(values))
                except (ValueError, TypeError):
                    # Handle non-numeric custom metrics
                    with open(self.run_dir / f"{name}.json", 'w') as f:
                        json.dump(values, f, indent=2)
        
        # Save training summary as JSON
        self._save_training_summary()
    
    def save_final_summary(self, final_episode: int, training_time: float) -> None:
        """
        Save final training summary with complete statistics.
        
        Args:
            final_episode: Final episode number
            training_time: Total training time in seconds
        """
        # Final save of all logs
        self.save_logs()
        
        # Get episode data for summary
        episode_rewards = []
        episode_lengths = []
        
        if self.env and hasattr(self.env, 'return_queue'):
            episode_rewards = list(self.env.return_queue)
        
        if self.env and hasattr(self.env, 'length_queue'):
            episode_lengths = list(self.env.length_queue)
        
        # Create comprehensive summary
        summary = {
            'training_info': {
                'total_episodes': final_episode + 1,
                'training_time_seconds': training_time,
                'training_time_minutes': training_time / 60,
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'episodes_per_second': (final_episode + 1) / training_time if training_time > 0 else 0
            },
            'configuration': self.training_config,
            'final_performance': self.get_recent_performance(min(100, len(episode_rewards)) if episode_rewards else 100),
            'training_progress': {
                'total_episodes': len(episode_rewards),
                'total_steps': sum(episode_lengths) if episode_lengths else 0
            }
        }
        
        # Add statistical analysis
        if episode_rewards:
            summary['reward_statistics'] = {
                'overall_mean': float(np.mean(episode_rewards)),
                'overall_std': float(np.std(episode_rewards)),
                'overall_max': float(np.max(episode_rewards)),
                'overall_min': float(np.min(episode_rewards)),
                'first_quartile': float(np.percentile(episode_rewards, 25)),
                'median': float(np.percentile(episode_rewards, 50)),
                'third_quartile': float(np.percentile(episode_rewards, 75))
            }
        
        if episode_lengths:
            summary['length_statistics'] = {
                'mean_episode_length': float(np.mean(episode_lengths)),
                'std_episode_length': float(np.std(episode_lengths)),
                'max_episode_length': int(np.max(episode_lengths)),
                'min_episode_length': int(np.min(episode_lengths))
            }
        
        # Save final summary
        with open(self.run_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create a concise results file for quick reference
        self._create_results_summary(summary)
    
    def _log_agent_metrics(self, agent: object) -> None:
        """Log agent-specific metrics."""
        # Log epsilon for exploration-based agents
        if hasattr(agent, 'epsilon'):
            self.epsilon_values.append(float(agent.epsilon))
        
        # Log learning rate if available
        if hasattr(agent, 'learning_rate'):
            self.learning_rates.append(float(agent.learning_rate))
        elif hasattr(agent, 'lr'):
            self.learning_rates.append(float(agent.lr))
        elif hasattr(agent, 'actor_lr'):
            # For actor-critic methods, log actor learning rate
            self.learning_rates.append(float(agent.actor_lr))
    
    def _save_training_summary(self) -> None:
        """Save intermediate training summary."""
        episode_rewards = []
        if self.env and hasattr(self.env, 'return_queue'):
            episode_rewards = list(self.env.return_queue)
        
        if not episode_rewards:
            return
        
        summary = {
            'episodes_completed': len(episode_rewards),
            'current_performance': self.get_recent_performance(50),  # Last 50 episodes
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.run_dir / "training_progress.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _create_results_summary(self, full_summary: Dict[str, Any]) -> None:
        """Create a concise results summary file."""
        if 'final_performance' not in full_summary:
            return
        
        perf = full_summary['final_performance']
        training_info = full_summary['training_info']
        
        results = [
            "=" * 60,
            "TRAINING RESULTS SUMMARY",
            "=" * 60,
            f"Total Episodes: {training_info['total_episodes']}",
            f"Training Time: {training_info['training_time_minutes']:.1f} minutes",
            f"Episodes/Second: {training_info['episodes_per_second']:.2f}",
            "",
            "FINAL PERFORMANCE (Last 100 episodes):",
            f"  Mean Reward: {perf.get('mean_reward', 'N/A'):.3f}" if 'mean_reward' in perf else "  Mean Reward: N/A",
            f"  Std Reward:  {perf.get('std_reward', 'N/A'):.3f}" if 'std_reward' in perf else "  Std Reward: N/A",
            f"  Max Reward:  {perf.get('max_reward', 'N/A'):.3f}" if 'max_reward' in perf else "  Max Reward: N/A",
            f"  Mean Length: {perf.get('mean_length', 'N/A'):.1f}" if 'mean_length' in perf else "  Mean Length: N/A",
            "",
            "=" * 60
        ]
        
        with open(self.run_dir / "results.txt", 'w') as f:
            f.write('\n'.join(results))
    
    def _load_existing_logs(self) -> None:
        """Load existing log files if they exist (for resuming training)."""
        try:
            # Load epsilon values
            epsilon_file = self.run_dir / "epsilon_values.npy"
            if epsilon_file.exists():
                self.epsilon_values = np.load(epsilon_file).tolist()
                print(f"Loaded {len(self.epsilon_values)} existing epsilon values")
            
            # Load learning rates
            lr_file = self.run_dir / "learning_rates.npy"
            if lr_file.exists():
                self.learning_rates = np.load(lr_file).tolist()
                print(f"Loaded {len(self.learning_rates)} existing learning rate values")
            
            # Load custom metrics
            for npy_file in self.run_dir.glob("*.npy"):
                if npy_file.name not in ["epsilon_values.npy", "learning_rates.npy", "episode_rewards.npy", "episode_lengths.npy", "training_errors.npy"]:
                    metric_name = npy_file.stem
                    try:
                        values = np.load(npy_file).tolist()
                        self.custom_metrics[metric_name] = values
                        print(f"Loaded {len(values)} existing {metric_name} values")
                    except Exception as e:
                        print(f"Warning: Could not load {npy_file}: {e}")
            
            # Also check for JSON files for non-numeric metrics
            for json_file in self.run_dir.glob("*.json"):
                if json_file.name not in ["config.yaml", "training_summary.json", "training_progress.json"]:
                    metric_name = json_file.stem
                    try:
                        with open(json_file, 'r') as f:
                            values = json.load(f)
                        if isinstance(values, list):  # Only load if it's a list of values
                            self.custom_metrics[metric_name] = values
                            print(f"Loaded {len(values)} existing {metric_name} values")
                    except Exception as e:
                        print(f"Warning: Could not load {json_file}: {e}")
                        
        except Exception as e:
            print(f"Warning: Error loading existing logs: {e}")