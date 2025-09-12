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
        
        # Initialize storage for various metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_errors = []
        self.custom_metrics = {}
        
        # Agent-specific metrics
        self.epsilon_values = []
        self.learning_rates = []
        
        # Timing information
        self.episode_times = []
        self.start_time = datetime.now()
        
        # Configuration
        self.save_frequency = training_config.get("log_frequency", 100)
        self.detailed_logging = training_config.get("detailed_logging", True)
    
    def log_episode(self, episode: int, env: gym.Env, agent: object) -> None:
        """
        Log statistics for a completed episode.
        
        Args:
            episode: Episode number
            env: Environment (with RecordEpisodeStatistics wrapper)
            agent: Agent that completed the episode
        """
        # Extract episode statistics from environment (optimized)
        if hasattr(env, 'return_queue') and len(env.return_queue) > 0:
            latest_reward = env.return_queue[-1]  # Direct access to last element
            self.episode_rewards.append(float(latest_reward))
        
        if hasattr(env, 'length_queue') and len(env.length_queue) > 0:
            latest_length = env.length_queue[-1]  # Direct access to last element
            self.episode_lengths.append(int(latest_length))
        
        # Log agent-specific metrics
        self._log_agent_metrics(agent)
        
        # Log custom metrics if agent provides them
        if hasattr(agent, 'get_metrics'):
            metrics = agent.get_metrics()
            for key, value in metrics.items():
                if key not in self.custom_metrics:
                    self.custom_metrics[key] = []
                self.custom_metrics[key].append(value)
        
        # Save logs periodically
        if (episode + 1) % self.save_frequency == 0:
            self.save_logs()
    
    def log_training_error(self, error: float) -> None:
        """
        Log training error (e.g., TD error, loss).
        
        Args:
            error: Training error value
        """
        self.training_errors.append(float(error))
    
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
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_errors': self.training_errors,
            'epsilon_values': self.epsilon_values,
            'learning_rates': self.learning_rates,
        }
        
        # Add custom metrics
        stats.update(self.custom_metrics)
        
        return stats
    
    def get_recent_performance(self, n_episodes: int = 100) -> Dict[str, float]:
        """
        Get performance metrics for recent episodes.
        
        Args:
            n_episodes: Number of recent episodes to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        performance = {}
        
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-n_episodes:]
            performance['mean_reward'] = float(np.mean(recent_rewards))
            performance['std_reward'] = float(np.std(recent_rewards))
            performance['max_reward'] = float(np.max(recent_rewards))
            performance['min_reward'] = float(np.min(recent_rewards))
        
        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-n_episodes:]
            performance['mean_length'] = float(np.mean(recent_lengths))
            performance['std_length'] = float(np.std(recent_lengths))
        
        if self.training_errors:
            recent_errors = self.training_errors[-n_episodes:]
            performance['mean_training_error'] = float(np.mean(recent_errors))
            performance['std_training_error'] = float(np.std(recent_errors))
        
        return performance
    
    def save_logs(self) -> None:
        """Save current logs to files."""
        # Save episode-wise data as numpy arrays
        if self.episode_rewards:
            np.save(self.run_dir / "episode_rewards.npy", np.array(self.episode_rewards))
        
        if self.episode_lengths:
            np.save(self.run_dir / "episode_lengths.npy", np.array(self.episode_lengths))
        
        if self.training_errors:
            np.save(self.run_dir / "training_errors.npy", np.array(self.training_errors))
        
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
            'final_performance': self.get_recent_performance(min(100, len(self.episode_rewards))),
            'training_progress': {
                'total_episodes': len(self.episode_rewards),
                'total_steps': sum(self.episode_lengths) if self.episode_lengths else 0
            }
        }
        
        # Add statistical analysis
        if self.episode_rewards:
            summary['reward_statistics'] = {
                'overall_mean': float(np.mean(self.episode_rewards)),
                'overall_std': float(np.std(self.episode_rewards)),
                'overall_max': float(np.max(self.episode_rewards)),
                'overall_min': float(np.min(self.episode_rewards)),
                'first_quartile': float(np.percentile(self.episode_rewards, 25)),
                'median': float(np.percentile(self.episode_rewards, 50)),
                'third_quartile': float(np.percentile(self.episode_rewards, 75))
            }
        
        if self.episode_lengths:
            summary['length_statistics'] = {
                'mean_episode_length': float(np.mean(self.episode_lengths)),
                'std_episode_length': float(np.std(self.episode_lengths)),
                'max_episode_length': int(np.max(self.episode_lengths)),
                'min_episode_length': int(np.min(self.episode_lengths))
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
        if not self.episode_rewards:
            return
        
        summary = {
            'episodes_completed': len(self.episode_rewards),
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