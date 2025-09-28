"""
Factory for creating agents based on configuration.
"""

import gymnasium as gym
from typing import Dict, Any, Union
from agents.q_learning import QLearning
from agents.adhdp import ADHDPAgent
from agents.actor_critic import ActorCritic
from agents.dqn import DQNAgent
from agents.idhp import IDHPAgent
from agents.td3 import TD3Agent
from agents.sac import SACAgent
from utils.discretizer import UniformTileCoding


def create_agent(agent_config: Dict[str, Any], env: gym.Env) -> object:
    """
    Create agent based on configuration and environment.
    
    Args:
        agent_config: Agent configuration dictionary
        env: Environment the agent will interact with
        
    Returns:
        Configured agent instance
        
    Raises:
        ValueError: If agent type is not supported
    """
    agent_type = agent_config["type"].lower()
    
    if agent_type == "q_learning" or agent_type == "qlearning":
        return create_q_learning_agent(agent_config, env)
    elif agent_type == "adhdp":
        return create_adhdp_agent(agent_config, env)
    elif agent_type == "actor_critic" or agent_type == "actorcritic":
        return create_actor_critic_agent(agent_config, env)
    elif agent_type == "dqn":
        return create_dqn_agent(agent_config, env)
    elif agent_type == "idhp":
        return create_idhp_agent(agent_config, env)
    elif agent_type == "td3":
        return create_td3_agent(agent_config, env)
    elif agent_type == "sac":
        return create_sac_agent(agent_config, env)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}. Supported types: q_learning, adhdp, actor_critic, dqn, idhp, td3, sac")


def create_q_learning_agent(agent_config: Dict[str, Any], env: gym.Env) -> QLearning:
    """
    Create Q-Learning agent.
    
    Args:
        agent_config: Agent configuration dictionary
        env: Environment the agent will interact with
        
    Returns:
        Configured Q-Learning agent
    """
    # Default parameters for Q-Learning
    defaults = {
        "learning_rate": 0.1,
        "initial_epsilon": 1.0,
        "epsilon_decay": None,
        "final_epsilon": 0.1,
        "discount_factor": 0.95
    }
    
    # Override defaults with config values
    params = {key: agent_config.get(key, default) for key, default in defaults.items()}
    
    # Handle epsilon decay calculation if not provided
    if params["epsilon_decay"] is None and "episodes" in agent_config:
        params["epsilon_decay"] = params["initial_epsilon"] / (agent_config["episodes"] / 2)
    elif params["epsilon_decay"] is None:
        params["epsilon_decay"] = 0.001  # Default fallback
    
    # Create discretizers if specified
    obs_discretizer = None
    action_discretizer = None
    
    if "obs_discretizer" in agent_config:
        obs_discretizer = create_discretizer(agent_config["obs_discretizer"], env.observation_space)
    
    if "action_discretizer" in agent_config:
        action_discretizer = create_discretizer(agent_config["action_discretizer"], env.action_space)
    
    # Create agent
    agent = QLearning(
        env=env,
        learning_rate=params["learning_rate"],
        initial_epsilon=params["initial_epsilon"],
        epsilon_decay=params["epsilon_decay"],
        final_epsilon=params["final_epsilon"],
        discount_factor=params["discount_factor"],
        obs_discretizer=obs_discretizer,
        action_discretizer=action_discretizer
    )
    
    return agent


def create_adhdp_agent(agent_config: Dict[str, Any], env: gym.Env) -> ADHDPAgent:
    """
    Create ADHDP agent.
    
    Args:
        agent_config: Agent configuration dictionary
        env: Environment the agent will interact with
        
    Returns:
        Configured ADHDP agent
    """
    # Default parameters for ADHDP
    defaults = {
        "obs_dim": env.observation_space.shape[0],
        "act_dim": env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1,
        "hidden_dim": 32,
        "num_layers": 2,
        "actor_lr": 0.001,
        "critic_lr": 0.001,
        "gamma": 0.99,
        "noise": True,
        "device": "cpu"
    }
    
    # Override defaults with config values
    params = {key: agent_config.get(key, default) for key, default in defaults.items()}
    
    # Extract action bounds from environment or config
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
        action_low = env.action_space.low
        action_high = env.action_space.high
    else:
        action_low = agent_config.get("action_low", [-1])
        action_high = agent_config.get("action_high", [1])
    
    # Create agent
    agent = ADHDPAgent(
        obs_dim=params["obs_dim"],
        act_dim=params["act_dim"],
        action_low=action_low,
        action_high=action_high,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"],
        gamma=params["gamma"],
        noise=params["noise"],
        device=params["device"]
    )
    
    return agent



def create_actor_critic_agent(agent_config: Dict[str, Any], env: gym.Env) -> ActorCritic:
    """
    Create Actor-Critic agent.
    
    Args:
        agent_config: Agent configuration dictionary
        env: Environment the agent will interact with
        
    Returns:
        Configured Actor-Critic agent
    """
    # Default parameters for Actor-Critic
    defaults = {
        "actor_lr": 0.001,
        "critic_lr": 0.001,
        "gamma": 0.99,
        "hidden_sizes": [64, 64]
    }
    
    # Override defaults with config values
    params = {key: agent_config.get(key, default) for key, default in defaults.items()}
    
    # Create agent
    agent = ActorCritic(
        env=env,
        discount_factor=params["gamma"],
        temperature_decay=0.999,
        final_temperature=0.01,
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"]
    )
    
    return agent


def create_discretizer(discretizer_config: Dict[str, Any], space: gym.Space) -> object:
    """
    Create discretizer based on configuration.
    
    Args:
        discretizer_config: Discretizer configuration dictionary
        space: Gymnasium space to discretize
        
    Returns:
        Configured discretizer
    """
    discretizer_type = discretizer_config.get("type", "uniform_tile_coding").lower()
    
    if discretizer_type == "uniform_tile_coding":
        bins = discretizer_config.get("bins", 10)
        return UniformTileCoding(space, bins=bins)
    else:
        raise ValueError(f"Unsupported discretizer type: {discretizer_type}")
    
def create_dqn_agent(agent_config: Dict[str, Any], env: gym.Env) -> DQNAgent:
    """
    Create DQN agent.
    
    Args:
        agent_config: Agent configuration dictionary
        env: Environment the agent will interact with
        
    Returns:
        Configured DQN agent
    """
    from agents.dqn import DQNAgent  # Import here to avoid circular dependencies
    
    # Default parameters for DQN
    defaults = {
        "hidden_sizes": [24, 24],
        "batch_size": 64,
        "memory_size": 10000,
        "learning_rate": 0.001,
        "discount_factor": 0.99,
        "tau": 0.01,
        "initial_epsilon": 1.0,
        "epsilon_decay": 0.995,
        "final_epsilon": 0.01,
        "device": "cpu"
    }
    
    # Override defaults with config values
    params = {key: agent_config.get(key, default) for key, default in defaults.items()}
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    
    # Create agent
    agent = DQNAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=params["hidden_sizes"],
        batch_size=params["batch_size"],
        memory_size=params["memory_size"],
        learning_rate=params["learning_rate"],
        gamma=params["discount_factor"],
        tau=params["tau"],
        epsilon_start=params["initial_epsilon"],
        epsilon_decay=params["epsilon_decay"],
        epsilon_min=params["final_epsilon"],
        device=params["device"],
        env=env
    )
    
    return agent


def create_idhp_agent(agent_config: Dict[str, Any], env: gym.Env) -> IDHPAgent:
    """
    Create IDHP agent.
    
    Args:
        agent_config: Agent configuration dictionary
        env: Environment the agent will interact with
        
    Returns:
        Configured IDHP agent
    """
    # Default parameters for IDHP
    defaults = {
        "hidden_sizes": [64, 64],
        "actor_lr": 0.0001,
        "critic_lr": 0.0002,
        "gamma": 0.99,
        "device": "cpu",
        "rls_lam": 0.99,
        "rls_delta": 1.0
    }
    
    # Override defaults with config values
    params = {key: agent_config.get(key, default) for key, default in defaults.items()}
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1
    
    # Create agent
    agent = IDHPAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=params["hidden_sizes"],
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"],
        gamma=params["gamma"],
        device=params["device"],
        env=env,
        rls_lam=params["rls_lam"],
        rls_delta=params["rls_delta"]
    )
    
    return agent


def create_td3_agent(agent_config: Dict[str, Any], env: gym.Env) -> TD3Agent:
    """
    Create TD3 agent.
    
    Args:
        agent_config: Agent configuration dictionary
        env: Environment the agent will interact with
        
    Returns:
        Configured TD3 agent
    """
    # Default parameters for TD3
    defaults = {
        "hidden_sizes": [256, 256],
        "batch_size": 64,
        "memory_size": 10000,
        "actor_lr": 0.001,
        "critic_lr": 0.001,
        "gamma": 0.99,
        "tau": 0.01,
        "exploration_noise_start": 0.1,
        "exploration_noise_decay": 0.95,
        "exploration_noise_min": 0.01,
        "target_noise_std": 0.2,
        "target_noise_clip": 0.5,
        "max_grad_norm": 1.0,
        "policy_delay": 2,
        "device": "cpu"
    }
    
    # Override defaults with config values
    params = {key: agent_config.get(key, default) for key, default in defaults.items()}
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1
    
    # Create agent
    agent = TD3Agent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=params["hidden_sizes"],
        batch_size=params["batch_size"],
        memory_size=params["memory_size"],
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"],
        gamma=params["gamma"],
        tau=params["tau"],
        exploration_noise_start=params["exploration_noise_start"],
        exploration_noise_decay=params["exploration_noise_decay"],
        exploration_noise_min=params["exploration_noise_min"],
        target_noise_std=params["target_noise_std"],
        target_noise_clip=params["target_noise_clip"],
        max_grad_norm=params["max_grad_norm"],
        policy_delay=params["policy_delay"],
        device=params["device"],
        env=env
    )
    
    return agent


def create_sac_agent(agent_config: Dict[str, Any], env: gym.Env) -> SACAgent:
    """
    Create SAC agent.
    
    Args:
        agent_config: Agent configuration dictionary
        env: Environment the agent will interact with
        
    Returns:
        Configured SAC agent
    """
    # Default parameters for SAC
    defaults = {
        "hidden_sizes": [256, 256],
        "batch_size": 256,
        "memory_size": 100000,
        "actor_lr": 0.0003,
        "critic_lr": 0.0003,
        "alpha_lr": 0.0003,
        "init_temp": 0.1,
        "gamma": 0.99,
        "tau": 0.005,
        "learnable_temp": True,
        "critic_target_update_freq": 1,
        "actor_update_freq": 1,
        "device": "cpu"
    }
    
    # Override defaults with config values
    params = {key: agent_config.get(key, default) for key, default in defaults.items()}
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1
    
    # Create agent
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=params["hidden_sizes"],
        batch_size=params["batch_size"],
        memory_size=params["memory_size"],
        actor_lr=params["actor_lr"],
        critic_lr=params["critic_lr"],
        alpha_lr=params["alpha_lr"],
        init_temp=params["init_temp"],
        gamma=params["gamma"],
        tau=params["tau"],
        learnable_temp=params["learnable_temp"],
        critic_target_update_freq=params["critic_target_update_freq"],
        actor_update_freq=params["actor_update_freq"],
        device=params["device"],
        env=env
    )
    
    return agent