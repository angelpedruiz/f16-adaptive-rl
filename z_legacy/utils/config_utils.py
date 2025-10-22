"""
Configuration utilities for loading and validating YAML configs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that the configuration contains all required sections and parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    required_sections = [
        "agent",
        "environment", 
        "training",
        "checkpointing",
        "plotting"
    ]
    
    # Check for required top-level sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate agent configuration
    agent_config = config["agent"]
    required_agent_params = ["type"]
    for param in required_agent_params:
        if param not in agent_config:
            raise ValueError(f"Missing required agent parameter: {param}")
    
    # Validate environment configuration
    env_config = config["environment"]
    required_env_params = ["name"]
    for param in required_env_params:
        if param not in env_config:
            raise ValueError(f"Missing required environment parameter: {param}")
    
    # Validate training configuration
    training_config = config["training"]
    required_training_params = ["episodes"]
    for param in required_training_params:
        if param not in training_config:
            raise ValueError(f"Missing required training parameter: {param}")
    
    if training_config["episodes"] <= 0:
        raise ValueError("Training episodes must be positive")
    
    # Validate checkpointing configuration
    checkpoint_config = config["checkpointing"]
    if "interval" in checkpoint_config:
        if checkpoint_config["interval"] <= 0:
            raise ValueError("Checkpoint interval must be positive")
    
    # Validate plotting configuration
    plotting_config = config["plotting"]
    if "training_metrics" in plotting_config:
        metrics_config = plotting_config["training_metrics"]
        if "interval" in metrics_config and metrics_config["interval"] <= 0:
            raise ValueError("Training metrics plotting interval must be positive")
    
    if "trajectory_episodes" in plotting_config:
        trajectory_episodes = plotting_config["trajectory_episodes"]
        if not isinstance(trajectory_episodes, list):
            raise ValueError("trajectory_episodes must be a list")
        
        max_episodes = training_config["episodes"]
        for ep in trajectory_episodes:
            if ep >= max_episodes:
                raise ValueError(f"Trajectory episode {ep} exceeds total episodes {max_episodes}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration dictionary with overrides
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_nested_config(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path to the configuration value (e.g., "agent.learning_rate")
        default: Default value to return if path doesn't exist
        
    Returns:
        Configuration value or default if not found
        
    Example:
        >>> config = {"agent": {"learning_rate": 0.001}}
        >>> get_nested_config(config, "agent.learning_rate")
        0.001
    """
    keys = path.split(".")
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except KeyError:
        return default


def set_nested_config(config: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary to modify
        path: Dot-separated path to the configuration value (e.g., "agent.learning_rate")
        value: Value to set
        
    Example:
        >>> config = {"agent": {}}
        >>> set_nested_config(config, "agent.learning_rate", 0.001)
        >>> config
        {"agent": {"learning_rate": 0.001}}
    """
    keys = path.split(".")
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value