import numpy as np
import gymnasium as gym
from pathlib import Path
from agents.q_learning import QLearning
from envs.f16_env import LinearModelF16
from utils.discretizer import UniformTileCoding
from utils.logging import setup_experiment_dir
from utils.checkpoint_utils import load_checkpoint
from data.LinearF16SS import A_long_hi_ref as A, B_long_hi as B
from utils.plots import plot_training_milestone_from_data
from utils.checkpoint_utils import save_checkpoint



def save_checkpoint_if_needed(agent, env, episode, checkpoint_interval, exp_dir, config):
    """Save a checkpoint every `checkpoint_interval` episodes."""
    if checkpoint_interval > 0 and episode != 0 and episode % checkpoint_interval == 0:
        checkpoint_dir = exp_dir / f"checkpoint_ep{episode}"
        save_checkpoint(
            agent=agent,
            current_episode=episode,
            return_queue=env.return_queue,
            length_queue=env.length_queue,
            config=config,
            checkpoint_dir=checkpoint_dir,
        )
        print(f"💾 Saved checkpoint at episode {episode}")

def record_milestone_if_needed(episode, milestones, info):
    """Initialize tracking structures if this episode is a milestone."""
    record_episode = episode in milestones
    if record_episode:
        print(f"📸 Recording milestone episode {episode}")
        state_trace = [np.copy(info["state"])]
        action_trace = []
        reward_trace = []
        reference_trace = [np.copy(info["reference"])]
        error_trace = [np.copy(info["tracking_error"])]
        return record_episode, state_trace, action_trace, reward_trace, reference_trace, error_trace
    return False, None, None, None, None, None

def save_milestone_plot(exp_dir, episode, state_trace, action_trace, reference_trace, error_trace, reward_trace, tracked_indices):
    """Plot and save milestone episode data."""
    milestone_path = exp_dir / f"state_action_evolution_ep{episode + 1}.png"
    plot_training_milestone_from_data(
        np.array(state_trace),
        np.array(action_trace),
        np.array(reference_trace),
        np.array(error_trace),
        reward_trace,
        episode_num=episode,
        save_path=milestone_path,
        tracked_indices=tracked_indices,
    )
    print(f"📈 Saved milestone plot for episode {episode}")

def save_final_metrics(agent, env, exp_dir, total_eps):
    """Save all post-training metrics and final checkpoint."""
    np.save(exp_dir / "returns.npy", np.array(env.return_queue))
    np.save(exp_dir / "lengths.npy", np.array(env.length_queue))
    np.save(exp_dir / "training_error.npy", np.array(agent.training_error))

    final_ckpt_dir = exp_dir / f"checkpoint_final_ep{total_eps}"
    save_checkpoint(
        agent=agent,
        current_episode=total_eps - 1,
        return_queue=env.return_queue,
        length_queue=env.length_queue,
        config=None,  # optional if you want to skip saving config here
        checkpoint_dir=final_ckpt_dir,
    )
    print(f"🏁 Saved final checkpoint at episode {total_eps}")
