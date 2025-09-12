"""
Factory for creating environments based on configuration.
"""

import gymnasium as gym
from typing import Dict, Any
from envs.f16_env import LinearModelF16
from data.LinearF16SS import A_long_hi_ref as A, B_long_hi as B


def create_environment(env_config: Dict[str, Any]) -> gym.Env:
    """
    Create environment based on configuration.
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        Configured gymnasium environment
        
    Raises:
        ValueError: If environment name is not supported
    """
    env_name = env_config["name"].lower()
    
    if env_name == "f16" or env_name == "linearmodelf16":
        return create_f16_environment(env_config)
    elif env_name in ["lunarlander", "lunar_lander"]:
        return create_lunar_lander_environment(env_config)
    elif env_name in ["invertedpendulum", "inverted_pendulum"]:
        return create_inverted_pendulum_environment(env_config)
    else:
        # Try to create a standard gymnasium environment
        return create_gym_environment(env_config)


def create_f16_environment(env_config: Dict[str, Any]) -> LinearModelF16:
    """
    Create F16 linear model environment.
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        Configured F16 environment
    """
    # Default parameters
    defaults = {
        "max_steps": 3000,
        "dt": 0.01,
        "state_indices_for_obs": [4],  # Keep only relevant states
        "reference_config": None,
        "action_low": [0, -22.5],
        "action_high": [0, 27],
        "obs_low": None,
        "obs_high": None
    }
    
    # Override defaults with config values
    for key, default_value in defaults.items():
        if key not in env_config:
            env_config[key] = default_value
    
    # Create environment
    env = LinearModelF16(
        A=A,
        B=B,
        max_steps=env_config["max_steps"],
        dt=env_config["dt"],
        reference_config=env_config.get("reference_config"),
        state_indices_for_obs=env_config["state_indices_for_obs"],
        action_low=env_config["action_low"],
        action_high=env_config["action_high"],
        obs_low=env_config.get("obs_low"),
        obs_high=env_config.get("obs_high")
    )
    
    return env


def create_lunar_lander_environment(env_config: Dict[str, Any]) -> gym.Env:
    """
    Create LunarLander environment.
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        Configured LunarLander environment
    """
    env_kwargs = {}
    
    # Add any LunarLander-specific configuration
    if "continuous" in env_config:
        env_name = "LunarLanderContinuous-v2" if env_config["continuous"] else "LunarLander-v2"
    else:
        env_name = "LunarLander-v2"
    
    if "max_episode_steps" in env_config:
        env_kwargs["max_episode_steps"] = env_config["max_episode_steps"]
    
    env = gym.make(env_name, **env_kwargs)
    
    # Apply any wrappers if specified
    if "wrappers" in env_config:
        env = apply_wrappers(env, env_config["wrappers"])
    
    return env


def create_inverted_pendulum_environment(env_config: Dict[str, Any]) -> gym.Env:
    """
    Create Inverted Pendulum environment.
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        Configured Inverted Pendulum environment
    """
    # Import the custom inverted pendulum environment
    from envs.inverted_pendulum_env import InvertedPendulumEnv
    
    env_kwargs = {}
    
    # Add configuration parameters
    for param in ["max_steps", "dt", "gravity", "length", "mass"]:
        if param in env_config:
            env_kwargs[param] = env_config[param]
    
    env = InvertedPendulumEnv(**env_kwargs)
    
    return env


def create_gym_environment(env_config: Dict[str, Any]) -> gym.Env:
    """
    Create a standard gymnasium environment.
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        Configured gymnasium environment
        
    Raises:
        ValueError: If environment name is not recognized by gymnasium
    """
    env_name = env_config["name"]
    env_kwargs = {}
    
    # Extract gymnasium-compatible parameters
    gym_params = ["max_episode_steps", "autoreset", "disable_env_checker"]
    for param in gym_params:
        if param in env_config:
            env_kwargs[param] = env_config[param]
    
    try:
        env = gym.make(env_name, **env_kwargs)
    except gym.error.Error as e:
        raise ValueError(f"Failed to create environment '{env_name}': {e}")
    
    # Apply any wrappers if specified
    if "wrappers" in env_config:
        env = apply_wrappers(env, env_config["wrappers"])
    
    return env


def apply_wrappers(env: gym.Env, wrapper_configs: list) -> gym.Env:
    """
    Apply a list of wrappers to an environment.
    
    Args:
        env: Base environment to wrap
        wrapper_configs: List of wrapper configurations
        
    Returns:
        Environment with applied wrappers
    """
    for wrapper_config in wrapper_configs:
        wrapper_name = wrapper_config["name"]
        wrapper_kwargs = wrapper_config.get("kwargs", {})
        
        # Map wrapper names to actual wrapper classes
        if wrapper_name == "FrameStack":
            env = gym.wrappers.FrameStack(env, **wrapper_kwargs)
        elif wrapper_name == "AtariPreprocessing":
            env = gym.wrappers.AtariPreprocessing(env, **wrapper_kwargs)
        elif wrapper_name == "NormalizeObservation":
            env = gym.wrappers.NormalizeObservation(env, **wrapper_kwargs)
        elif wrapper_name == "NormalizeReward":
            env = gym.wrappers.NormalizeReward(env, **wrapper_kwargs)
        elif wrapper_name == "ClipAction":
            env = gym.wrappers.ClipAction(env, **wrapper_kwargs)
        elif wrapper_name == "RescaleAction":
            env = gym.wrappers.RescaleAction(env, **wrapper_kwargs)
        else:
            # Try to get wrapper from gym.wrappers
            try:
                wrapper_class = getattr(gym.wrappers, wrapper_name)
                env = wrapper_class(env, **wrapper_kwargs)
            except AttributeError:
                raise ValueError(f"Unknown wrapper: {wrapper_name}")
    
    return env