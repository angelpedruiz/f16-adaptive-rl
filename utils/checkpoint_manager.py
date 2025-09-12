"""
Checkpoint management for saving and loading agent states and training progress.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym


class CheckpointManager:
    """
    Manages checkpointing and resuming of training sessions.
    
    Handles saving/loading of:
    - Agent parameters (brain, discretizers, normalization)
    - Training statistics (episode rewards, lengths, errors)
    - Environment state tracking
    - Configuration snapshots
    """
    
    def __init__(self, run_dir: Path, checkpoint_config: Dict[str, Any], agent_type: str):
        """
        Initialize checkpoint manager.
        
        Args:
            run_dir: Directory for this training run
            checkpoint_config: Checkpointing configuration
            agent_type: Type of agent being trained
        """
        self.run_dir = Path(run_dir)
        self.checkpoint_config = checkpoint_config
        self.agent_type = agent_type.lower()
        
        # Extract checkpoint settings
        self.checkpoint_interval = checkpoint_config.get("interval", 1000)
        self.keep_last_n = checkpoint_config.get("keep_last_n", 5)
        self.save_best = checkpoint_config.get("save_best", True)
        
        # Track best performance for saving best checkpoint
        self.best_reward = float('-inf')
        self.best_episode = 0
        
        # Keep track of saved checkpoints for cleanup
        self.saved_checkpoints = []
    
    def should_save_checkpoint(self, episode: int) -> bool:
        """
        Determine if a checkpoint should be saved at this episode.
        
        Args:
            episode: Current episode number
            
        Returns:
            True if checkpoint should be saved
        """
        return (episode + 1) % self.checkpoint_interval == 0 or episode == 0
    
    def save_checkpoint(self, episode: int, agent: object, env: gym.Env, 
                       training_logger: object, is_final: bool = False) -> str:
        """
        Save training checkpoint including agent state and training statistics.
        
        Args:
            episode: Current episode number
            agent: Agent to save
            env: Environment (for statistics)
            training_logger: Training logger with statistics
            is_final: Whether this is the final checkpoint
            
        Returns:
            Path to saved checkpoint directory
        """
        # Create checkpoint directory
        if is_final:
            checkpoint_name = f"checkpoint_final_ep{episode + 1}"
        else:
            checkpoint_name = f"checkpoint_ep{episode}"
        
        checkpoint_dir = self.run_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save agent state
        self._save_agent_state(agent, checkpoint_dir, episode)
        
        # Save training statistics
        self._save_training_stats(env, training_logger, checkpoint_dir)
        
        # Save checkpoint metadata
        self._save_checkpoint_metadata(episode, checkpoint_dir, is_final)
        
        # Add to tracked checkpoints
        self.saved_checkpoints.append(checkpoint_dir)
        
        # Clean up old checkpoints if needed
        if not is_final:
            self._cleanup_old_checkpoints()
        
        # Check if this is the best checkpoint
        if self.save_best and hasattr(env, 'return_queue') and len(env.return_queue) > 0:
            recent_reward = np.mean(list(env.return_queue)[-10:])  # Average of last 10 episodes
            if recent_reward > self.best_reward:
                self.best_reward = recent_reward
                self.best_episode = episode
                self._save_best_checkpoint(checkpoint_dir)
        
        return str(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint_path: str, agent: object, env: gym.Env) -> int:
        """
        Load training checkpoint and restore agent state.
        
        Args:
            checkpoint_path: Path to checkpoint file or directory
            agent: Agent to restore state to
            env: Environment (for statistics restoration)
            
        Returns:
            Episode number to resume training from
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Handle both file and directory paths
        if checkpoint_path.is_file():
            checkpoint_dir = checkpoint_path.parent
            agent_file = checkpoint_path
        else:
            checkpoint_dir = checkpoint_path
            agent_file = self._find_agent_file(checkpoint_dir)
        
        # Load agent state
        resume_episode = self._load_agent_state(agent, agent_file)
        
        # Load training statistics if available
        self._load_training_stats(env, checkpoint_dir)
        
        return resume_episode + 1  # Resume from next episode
    
    def _save_agent_state(self, agent: object, checkpoint_dir: Path, episode: int) -> None:
        """Save agent-specific state based on agent type."""
        if hasattr(agent, 'save'):
            # Agent has custom save method
            agent.save(checkpoint_dir / f"checkpoint_ep{episode}.pkl")
        else:
            # Default saving based on agent type
            if self.agent_type in ['q_learning', 'qlearning']:
                self._save_q_learning_agent(agent, checkpoint_dir, episode)
            elif self.agent_type == 'adhdp':
                self._save_adhdp_agent(agent, checkpoint_dir, episode)
            elif self.agent_type == 'dqn':
                self._save_dqn_agent(agent, checkpoint_dir, episode)
            else:
                # Generic save using pickle
                with open(checkpoint_dir / f"checkpoint_ep{episode}.pkl", 'wb') as f:
                    pickle.dump(agent, f)
    
    def _save_q_learning_agent(self, agent: object, checkpoint_dir: Path, episode: int) -> None:
        """Save Q-Learning agent state."""
        save_dict = {
            'episode': episode,
            'epsilon': getattr(agent, 'epsilon', None),
            'q_table': getattr(agent, 'q_table', None),
        }
        
        # Save discretizers if they exist
        if hasattr(agent, 'obs_discretizer'):
            save_dict['obs_discretizer'] = agent.obs_discretizer
        if hasattr(agent, 'action_discretizer'):
            save_dict['action_discretizer'] = agent.action_discretizer
        
        # Save as npz file
        np.savez(
            checkpoint_dir / f"checkpoint_ep{episode}_brain.npz",
            **{k: v for k, v in save_dict.items() if v is not None}
        )
        
        # Also save as JSON for readability
        json_dict = {k: v for k, v in save_dict.items() 
                    if k not in ['q_table', 'obs_discretizer', 'action_discretizer']}
        with open(checkpoint_dir / f"checkpoint_ep{episode}.json", 'w') as f:
            json.dump(json_dict, f, indent=2)
    
    def _save_adhdp_agent(self, agent: object, checkpoint_dir: Path, episode: int) -> None:
        """Save ADHDP agent state."""
        save_dict = {
            'episode': episode,
        }
        
        # Save PyTorch model parameters if available
        try:
            import torch
            if hasattr(agent, 'actor') and hasattr(agent.actor, 'state_dict'):
                torch.save(agent.actor.state_dict(), 
                          checkpoint_dir / f"actor_ep{episode}.pth")
            if hasattr(agent, 'critic') and hasattr(agent.critic, 'state_dict'):
                torch.save(agent.critic.state_dict(), 
                          checkpoint_dir / f"critic_ep{episode}.pth")
        except ImportError:
            pass
        
        # Save other parameters
        if hasattr(agent, 'get_parameters'):
            params = agent.get_parameters()
            np.savez(checkpoint_dir / f"checkpoint_ep{episode}_brain.npz", **params)
        else:
            # Fallback to pickle
            with open(checkpoint_dir / f"checkpoint_ep{episode}.pkl", 'wb') as f:
                pickle.dump(agent, f)
    
    def _save_dqn_agent(self, agent: object, checkpoint_dir: Path, episode: int) -> None:
        """Save DQN agent state."""
        try:
            import torch
            if hasattr(agent, 'q_network') and hasattr(agent.q_network, 'state_dict'):
                torch.save(agent.q_network.state_dict(), 
                          checkpoint_dir / f"q_network_ep{episode}.pth")
            if hasattr(agent, 'target_network') and hasattr(agent.target_network, 'state_dict'):
                torch.save(agent.target_network.state_dict(), 
                          checkpoint_dir / f"target_network_ep{episode}.pth")
        except ImportError:
            pass
        
        # Save other parameters
        save_dict = {
            'episode': episode,
            'epsilon': getattr(agent, 'epsilon', None),
            'memory_size': getattr(agent, 'memory_size', None),
        }
        
        with open(checkpoint_dir / f"checkpoint_ep{episode}.json", 'w') as f:
            json.dump({k: v for k, v in save_dict.items() if v is not None}, f, indent=2)
    
    def _save_training_stats(self, env: gym.Env, training_logger: object, 
                           checkpoint_dir: Path) -> None:
        """Save training statistics."""
        # Save episode statistics from RecordEpisodeStatistics wrapper
        if hasattr(env, 'return_queue') and len(env.return_queue) > 0:
            np.save(checkpoint_dir / "returns.npy", np.array(list(env.return_queue)))
        
        if hasattr(env, 'length_queue') and len(env.length_queue) > 0:
            np.save(checkpoint_dir / "lengths.npy", np.array(list(env.length_queue)))
        
        # Save training logger statistics
        if hasattr(training_logger, 'get_training_stats'):
            stats = training_logger.get_training_stats()
            for key, values in stats.items():
                if isinstance(values, (list, np.ndarray)):
                    np.save(checkpoint_dir / f"{key}.npy", np.array(values))
    
    def _save_checkpoint_metadata(self, episode: int, checkpoint_dir: Path, 
                                is_final: bool) -> None:
        """Save metadata about this checkpoint."""
        metadata = {
            'episode': episode,
            'agent_type': self.agent_type,
            'is_final': is_final,
            'timestamp': str(Path().resolve()),
            'checkpoint_config': self.checkpoint_config
        }
        
        with open(checkpoint_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_agent_state(self, agent: object, agent_file: Path) -> int:
        """Load agent state from file."""
        if hasattr(agent, 'load'):
            return agent.load(agent_file)
        
        # Default loading based on file extension
        if agent_file.suffix == '.npz':
            return self._load_npz_agent(agent, agent_file)
        elif agent_file.suffix == '.pkl':
            return self._load_pickle_agent(agent, agent_file)
        elif agent_file.suffix == '.json':
            return self._load_json_agent(agent, agent_file)
        else:
            raise ValueError(f"Unsupported checkpoint file format: {agent_file.suffix}")
    
    def _load_npz_agent(self, agent: object, agent_file: Path) -> int:
        """Load agent from npz file."""
        data = np.load(agent_file, allow_pickle=True)
        
        episode = int(data.get('episode', 0))
        
        # Restore agent parameters
        for key in data.files:
            if hasattr(agent, key):
                setattr(agent, key, data[key].item() if data[key].ndim == 0 else data[key])
        
        return episode
    
    def _load_pickle_agent(self, agent: object, agent_file: Path) -> int:
        """Load agent from pickle file."""
        with open(agent_file, 'rb') as f:
            saved_agent = pickle.load(f)
        
        # Copy attributes from saved agent
        for attr_name in dir(saved_agent):
            if not attr_name.startswith('_'):
                setattr(agent, attr_name, getattr(saved_agent, attr_name))
        
        return getattr(saved_agent, 'episode', 0)
    
    def _load_json_agent(self, agent: object, agent_file: Path) -> int:
        """Load agent from JSON file."""
        with open(agent_file, 'r') as f:
            data = json.load(f)
        
        episode = data.get('episode', 0)
        
        # Restore simple parameters
        for key, value in data.items():
            if hasattr(agent, key) and key != 'episode':
                setattr(agent, key, value)
        
        return episode
    
    def _load_training_stats(self, env: gym.Env, checkpoint_dir: Path) -> None:
        """Load training statistics if available."""
        # This is more complex with RecordEpisodeStatistics wrapper
        # For now, just check if files exist
        returns_file = checkpoint_dir / "returns.npy"
        lengths_file = checkpoint_dir / "lengths.npy"
        
        if returns_file.exists() and lengths_file.exists():
            # Files exist but restoring to wrapper is complex
            # Could implement if needed for exact resume functionality
            pass
    
    def _find_agent_file(self, checkpoint_dir: Path) -> Path:
        """Find the agent file in checkpoint directory."""
        # Look for common agent file patterns
        patterns = [
            "*_brain.npz",
            "checkpoint_*.npz", 
            "checkpoint_*.pkl",
            "checkpoint_*.json"
        ]
        
        for pattern in patterns:
            files = list(checkpoint_dir.glob(pattern))
            if files:
                return files[0]  # Return first match
        
        raise FileNotFoundError(f"No agent file found in {checkpoint_dir}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to save disk space."""
        if len(self.saved_checkpoints) > self.keep_last_n:
            # Sort by episode number
            self.saved_checkpoints.sort(key=lambda x: x.name)
            
            # Remove oldest checkpoints
            to_remove = self.saved_checkpoints[:-self.keep_last_n]
            for checkpoint_dir in to_remove:
                if checkpoint_dir.exists() and "final" not in checkpoint_dir.name:
                    try:
                        import shutil
                        shutil.rmtree(checkpoint_dir)
                        self.saved_checkpoints.remove(checkpoint_dir)
                    except OSError:
                        pass  # Failed to remove, keep trying
    
    def _save_best_checkpoint(self, current_checkpoint_dir: Path) -> None:
        """Save a copy of the best checkpoint."""
        best_dir = self.run_dir / "checkpoint_best"
        
        # Remove previous best if exists
        if best_dir.exists():
            import shutil
            shutil.rmtree(best_dir)
        
        # Copy current checkpoint as best
        import shutil
        shutil.copytree(current_checkpoint_dir, best_dir)