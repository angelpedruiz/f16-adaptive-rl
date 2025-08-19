import numpy as np
import gymnasium as gym
from pathlib import Path
from agents.q_learning import QLearning
from envs.f16_env import LinearModelF16
from utils.discretizer import UniformTileCoding
from utils.logging import setup_experiment_dir
from utils.checkpoint_utils import load_checkpoint
from data.LinearF16SS import A_long_hi_ref as A, B_long_hi as B, A_f1
from utils.plots import plot_training_milestone_from_data
from utils.checkpoint_utils import save_checkpoint

def apply_fault(env, fault_type):
    if fault_type == "elevator_loss":
        env.A = A_f1 
    else:
        raise ValueError(f"Unknown fault type: {fault_type}")

def setup_training(config):
    """
    Sets up the environment, agent, experiment folder, and optionally resumes from a checkpoint.
    
    Returns:
        env: Gym environment
        agent: QLearning agent
        exp_dir: Path to experiment directory
        start_episode: int, starting episode number
        total_eps: int, total episodes to train
    """
    # Set random seed
    np.random.seed(config["training"]["seed"])

    # Create environment
    env = LinearModelF16(
        A,
        B,
        max_steps=config["env"]["max_steps"],
        dt=config["env"]["dt"],
        reference_config=config["env"]["reference_config"],
        state_indices_for_obs=config["env"]["state_indices_to_keep"],
        action_low=config["env"]["action_low"],
        action_high=config["env"]["action_high"],
        obs_low=config["env"]["obs_low"],
        obs_high=config["env"]["obs_high"],
    )
    
    fault_type = config["fault"]["fault_type"]
    if fault_type is not None:  # Only apply if specified
        apply_fault(env, fault_type)
        
    env = gym.wrappers.RecordEpisodeStatistics(
        env, buffer_length=config["training"]["episodes"]
    )

    # Create agent
    agent = QLearning(
        env=env,
        learning_rate=config["agent"]["learning_rate"],
        initial_epsilon=config["agent"]["epsilon"]["start"],
        epsilon_decay=config["agent"]["epsilon"]["decay"],
        final_epsilon=config["agent"]["epsilon"]["final"],
        discount_factor=config["agent"]["discount_factor"],
        obs_discretizer=UniformTileCoding(env.observation_space, config["agent"]["obs_bins"]),
        action_discretizer=UniformTileCoding(env.action_space, config["agent"]["action_bins"]),
    )

    # Setup experiment folder / resume
    resume_from = config["training"].get("resume_from", None)
    if resume_from:
        ckpt_path = Path(resume_from).resolve()
        exp_dir = ckpt_path.parent.parent
        ckpt = load_checkpoint(ckpt_path)
        agent.load_brain(ckpt["agent_brain"])
        agent.obs_discretizer.set_params(ckpt["obs_discretizer"])
        agent.action_discretizer.set_params(ckpt["action_discretizer"])
        agent.training_error = ckpt["training_error"].tolist()
        env.return_queue = ckpt["returns"].tolist()
        env.length_queue = ckpt["lengths"].tolist()
        start_episode = ckpt["episode"] + 1
        print(f"🔄 Resuming training from checkpoint: {resume_from}")
        print(f"   Episodes completed: {start_episode}/{config['training']['episodes']}")
        total_eps = config["training"]["episodes"] + start_episode
    else:
        exp_dir = setup_experiment_dir(config, algo_name="q_learning")
        start_episode = 0
        total_eps = config["training"]["episodes"]

    return env, agent, exp_dir, start_episode, total_eps, fault_type


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
