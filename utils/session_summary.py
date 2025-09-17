"""
Session summary generator for training runs.
Creates comprehensive .txt summaries for each training session.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import gymnasium as gym


def create_session_summary(run_dir: Path, config: dict, final_episode: int, start_episode: int,
                          session_time: float, resume_from: Optional[str], env: gym.Env, 
                          agent: object, training_logger: object) -> None:
    """
    Create a comprehensive session summary in .txt format.
    
    Args:
        run_dir: Run directory path
        config: Training configuration
        final_episode: Final episode number
        start_episode: Starting episode number for this session
        session_time: Training time for this session
        resume_from: Checkpoint path used to resume (None for new sessions)
        env: Environment with episode statistics
        agent: Trained agent
        training_logger: Logger with training metrics
    """
    # Calculate session information
    episodes_completed = max(final_episode + 1 - start_episode, 0)
    total_episodes_in_run = final_episode + 1
    
    # Load or initialize cumulative time tracking
    cumulative_time_file = run_dir / "cumulative_training_time.json"
    cumulative_data = {}
    
    if cumulative_time_file.exists():
        try:
            with open(cumulative_time_file, 'r') as f:
                cumulative_data = json.load(f)
        except:
            cumulative_data = {}
    
    # Update cumulative time
    previous_time = cumulative_data.get('total_training_time', 0.0)
    total_training_time = previous_time + session_time
    
    cumulative_data.update({
        'total_training_time': total_training_time,
        'total_sessions': cumulative_data.get('total_sessions', 0) + 1,
        'last_session_time': session_time,
        'last_update': datetime.now().isoformat()
    })
    
    # Save updated cumulative data
    with open(cumulative_time_file, 'w') as f:
        json.dump(cumulative_data, f, indent=2)
    
    # Get performance metrics
    performance = {}
    if hasattr(env, 'return_queue') and len(env.return_queue) > 0:
        rewards = list(env.return_queue)
        session_rewards = rewards[start_episode:] if len(rewards) > start_episode else rewards
        if session_rewards:
            performance['session_mean_reward'] = np.mean(session_rewards)
            performance['session_total_reward'] = np.sum(session_rewards)
            performance['overall_mean_reward'] = np.mean(rewards)
        
    if hasattr(env, 'length_queue') and len(env.length_queue) > 0:
        lengths = list(env.length_queue)
        session_lengths = lengths[start_episode:] if len(lengths) > start_episode else lengths
        if session_lengths:
            performance['session_total_steps'] = np.sum(session_lengths)
            performance['overall_total_steps'] = np.sum(lengths)
            performance['session_mean_steps'] = np.mean(session_lengths)
    
    # Create comprehensive summary
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("TRAINING SESSION SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    
    # Session Information
    _add_session_info(summary_lines, resume_from, start_episode, final_episode, 
                     episodes_completed, total_episodes_in_run, session_time, 
                     total_training_time, cumulative_data)
    
    # Agent Configuration
    _add_agent_config(summary_lines, config.get("agent", {}))
    
    # Environment Configuration
    _add_environment_config(summary_lines, config.get("environment", {}))
    
    # Training Configuration
    _add_training_config(summary_lines, config.get("training", {}))
    
    # Performance Metrics
    _add_performance_metrics(summary_lines, performance, agent)
    
    # Additional Information
    _add_additional_info(summary_lines, episodes_completed, session_time, run_dir)
    
    summary_lines.append("=" * 80)
    
    # Save session summary
    session_num = cumulative_data['total_sessions']
    summary_filename = f"session_{session_num:03d}_summary.txt"
    summary_path = run_dir / summary_filename
    
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Session summary saved to: {summary_filename}")


def _add_session_info(summary_lines: list, resume_from: Optional[str], start_episode: int,
                     final_episode: int, episodes_completed: int, total_episodes_in_run: int,
                     session_time: float, total_training_time: float, cumulative_data: dict) -> None:
    """Add session information to summary."""
    summary_lines.append("SESSION INFORMATION:")
    summary_lines.append("-" * 40)
    if resume_from:
        summary_lines.append(f"Session Type:        Resumed Training")
        summary_lines.append(f"Resumed From:        {resume_from}")
        summary_lines.append(f"Start Episode:       {start_episode}")
    else:
        summary_lines.append(f"Session Type:        New Training")
        summary_lines.append(f"Start Episode:       0")
    
    summary_lines.append(f"End Episode:         {final_episode}")
    summary_lines.append(f"Episodes This Session: {episodes_completed}")
    summary_lines.append(f"Total Episodes in Run: {total_episodes_in_run}")
    summary_lines.append(f"Session Training Time: {session_time/60:.2f} minutes ({session_time:.1f} seconds)")
    summary_lines.append(f"Total Run Time:      {total_training_time/60:.2f} minutes ({total_training_time:.1f} seconds)")
    summary_lines.append(f"Total Sessions:      {cumulative_data['total_sessions']}")
    summary_lines.append("")


def _add_agent_config(summary_lines: list, agent_config: dict) -> None:
    """Add agent configuration to summary."""
    summary_lines.append("AGENT CONFIGURATION:")
    summary_lines.append("-" * 40)
    summary_lines.append(f"Agent Type:          {agent_config.get('type', 'Unknown')}")
    
    if 'learning_rate' in agent_config:
        summary_lines.append(f"Learning Rate:       {agent_config['learning_rate']}")
    if 'initial_epsilon' in agent_config:
        summary_lines.append(f"Initial Epsilon:     {agent_config['initial_epsilon']}")
    if 'final_epsilon' in agent_config:
        summary_lines.append(f"Final Epsilon:       {agent_config['final_epsilon']}")
    if 'discount_factor' in agent_config:
        summary_lines.append(f"Discount Factor:     {agent_config['discount_factor']}")
    
    if 'obs_discretizer' in agent_config:
        obs_disc = agent_config['obs_discretizer']
        summary_lines.append(f"Obs Discretizer:     {obs_disc.get('type', 'Unknown')} - {obs_disc.get('bins', 'N/A')} bins")
    
    if 'action_discretizer' in agent_config:
        act_disc = agent_config['action_discretizer']
        summary_lines.append(f"Action Discretizer:  {act_disc.get('type', 'Unknown')} - {act_disc.get('bins', 'N/A')} bins")
    
    summary_lines.append("")


def _add_environment_config(summary_lines: list, env_config: dict) -> None:
    """Add environment configuration to summary."""
    summary_lines.append("ENVIRONMENT CONFIGURATION:")
    summary_lines.append("-" * 40)
    summary_lines.append(f"Environment:         {env_config.get('name', 'Unknown')}")
    summary_lines.append(f"Max Steps:           {env_config.get('max_steps', 'N/A')}")
    summary_lines.append(f"Time Step (dt):      {env_config.get('dt', 'N/A')}")
    summary_lines.append(f"Action Bounds:       [{env_config.get('action_low', 'N/A')}, {env_config.get('action_high', 'N/A')}]")
    summary_lines.append(f"Observation Bounds:  [{env_config.get('obs_low', 'N/A')}, {env_config.get('obs_high', 'N/A')}]")
    
    ref_config = env_config.get('reference_config', {})
    if ref_config:
        summary_lines.append("Reference Configurations:")
        for state_idx, ref_cfg in ref_config.items():
            ref_type = ref_cfg.get('type', 'Unknown')
            summary_lines.append(f"  State {state_idx}: {ref_type}")
            
            if ref_type == "sin":
                summary_lines.append(f"    Amplitude: {ref_cfg.get('A', 'N/A')}")
                summary_lines.append(f"    Period: {ref_cfg.get('T', 'N/A')}s")
                if 'phi' in ref_cfg:
                    summary_lines.append(f"    Phase: {ref_cfg.get('phi')} rad")
            elif ref_type == "constant":
                summary_lines.append(f"    Value: {ref_cfg.get('value', 'N/A')}")
            elif ref_type == "cos_step":
                amp = ref_cfg.get('amplitude', {})
                step_dur = ref_cfg.get('step_duration', {})
                trans_dur = ref_cfg.get('transition_duration', {})
                summary_lines.append(f"    Amplitude: [{amp.get('min', 'N/A')}, {amp.get('max', 'N/A')}], {amp.get('n_levels', 'N/A')} levels")
                summary_lines.append(f"    Step Duration: [{step_dur.get('min', 'N/A')}, {step_dur.get('max', 'N/A')}]s")
                summary_lines.append(f"    Transition Duration: [{trans_dur.get('min', 'N/A')}, {trans_dur.get('max', 'N/A')}]s")
    
    summary_lines.append("")


def _add_training_config(summary_lines: list, training_config: dict) -> None:
    """Add training configuration to summary."""
    summary_lines.append("TRAINING CONFIGURATION:")
    summary_lines.append("-" * 40)
    summary_lines.append(f"Total Episodes:      {training_config.get('episodes', 'N/A')}")
    summary_lines.append(f"Random Seed:         {training_config.get('seed', 'N/A')}")
    summary_lines.append(f"Log Frequency:       {training_config.get('log_frequency', 'N/A')}")
    summary_lines.append("")


def _add_performance_metrics(summary_lines: list, performance: dict, agent: object) -> None:
    """Add performance metrics to summary."""
    summary_lines.append("PERFORMANCE METRICS:")
    summary_lines.append("-" * 40)
    
    if 'session_mean_reward' in performance:
        summary_lines.append(f"Session Mean Reward:   {performance['session_mean_reward']:.4f}")
        summary_lines.append(f"Session Total Reward:  {performance['session_total_reward']:.2f}")
        summary_lines.append(f"Overall Mean Reward:   {performance['overall_mean_reward']:.4f}")
    
    if 'session_total_steps' in performance:
        summary_lines.append(f"Session Total Steps:   {performance['session_total_steps']:,}")
        summary_lines.append(f"Session Mean Steps:    {performance['session_mean_steps']:.1f}")
        summary_lines.append(f"Overall Total Steps:   {performance['overall_total_steps']:,}")
    
    # Agent-specific metrics
    if hasattr(agent, 'epsilon'):
        summary_lines.append(f"Final Epsilon:         {agent.epsilon:.4f}")
    
    if hasattr(agent, 'training_error') and agent.training_error:
        if isinstance(agent.training_error, list):
            latest_error = agent.training_error[-1] if agent.training_error else 0
        else:
            latest_error = agent.training_error
        summary_lines.append(f"Latest Training Error: {latest_error:.6f}")
    
    summary_lines.append("")


def _add_additional_info(summary_lines: list, episodes_completed: int, session_time: float, run_dir: Path) -> None:
    """Add additional information to summary."""
    summary_lines.append("ADDITIONAL INFORMATION:")
    summary_lines.append("-" * 40)
    
    if episodes_completed > 0:
        avg_time_per_episode = session_time / episodes_completed
        summary_lines.append(f"Avg Time per Episode:  {avg_time_per_episode:.3f} seconds")
        episodes_per_minute = episodes_completed / (session_time / 60) if session_time > 0 else 0
        summary_lines.append(f"Episodes per Minute:   {episodes_per_minute:.2f}")
    
    summary_lines.append(f"Run Directory:         {run_dir}")
    summary_lines.append("")