# F16 Adaptive Reinforcement Learning - Refactored

A modular, scalable, and fully config-driven reinforcement learning framework for F16 aircraft control tasks.

## 🏗️ Project Structure

```
f16-adaptive-rl/
├── train.py                 # Unified training script
├── example_usage.py         # Usage examples and documentation
├── agents/                  # Agent implementations
│   ├── q_learning.py       # Q-Learning agent
│   ├── adhdp.py           # ADHDP agent
│   ├── dqn.py             # Deep Q-Network agent
│   ├── td3.py             # TD3 agent
│   └── base_agent.py      # Base agent class
├── envs/                   # Environment implementations
│   ├── f16_env.py         # F16 linear model environment
│   └── inverted_pendulum_env.py
├── utils/                  # Utility modules
│   ├── config_utils.py    # Configuration loading and validation
│   ├── environment_factory.py  # Environment creation
│   ├── agent_factory.py   # Agent creation
│   ├── checkpoint_manager.py   # Checkpointing system
│   ├── plotting_manager.py     # Plot generation
│   ├── training_logger.py      # Training statistics
│   └── ...                # Other utilities
├── configs/               # Configuration templates
│   ├── adhdp_config.yaml  # ADHDP configuration
│   ├── q_learning_config.yaml  # Q-Learning configuration
│   ├── dqn_config.yaml    # DQN configuration
│   └── test_config.yaml   # Minimal test configuration
└── experiments/           # Training results (auto-generated)
    └── [agent_name]/
        └── run_[timestamp]/
            ├── config.yaml
            ├── checkpoint_*.npz
            ├── *.png
            └── training_summary.json
```

## 🚀 Quick Start

### 1. Basic Training

Train an ADHDP agent:
```bash
python train.py --config configs/adhdp_config.yaml
```

Train a Q-Learning agent:
```bash
python train.py --config configs/q_learning_config.yaml
```

### 2. Resume Training

Resume from a previous checkpoint:
```bash
python train.py --config configs/adhdp_config.yaml --resume-from experiments/adhdp/run_20241201_123456/checkpoint_ep1000
```

### 3. View Examples

See all usage examples:
```bash
python example_usage.py
```

## ⚙️ Configuration System

All training parameters are defined in YAML configuration files with the following structure:

### Agent Configuration
```yaml
agent:
  type: "ADHDP"              # Agent type
  obs_dim: 2                 # Observation dimension
  act_dim: 2                 # Action dimension
  learning_rate: 0.001       # Learning rate
  # ... other agent-specific parameters
```

### Environment Configuration
```yaml
environment:
  name: "f16"                # Environment name
  max_steps: 3000           # Episode length
  dt: 0.01                  # Time step
  reference_config:         # Reference signal configuration
    1:
      type: "sin"           # Signal type
      A: 0.35               # Amplitude
      T: 5.0                # Period
```

### Training Configuration
```yaml
training:
  episodes: 1000            # Total episodes
  seed: 42                  # Random seed
  log_frequency: 100        # Logging interval
```

### Checkpointing Configuration
```yaml
checkpointing:
  interval: 200             # Save every N episodes
  keep_last_n: 5           # Keep last N checkpoints
  save_best: true          # Save best performing model
  resume_from: null        # Resume checkpoint path
```

### Plotting Configuration
```yaml
plotting:
  training_metrics:
    interval: 100           # Plot every N episodes
    rolling_window: 50      # Moving average window
    save_individual: true   # Save individual plots
  
  trajectories:
    episodes: [100, 500, 1000]  # Episodes to plot
    save_data: true         # Save trajectory data
```

## 🎯 Key Features

### ✅ Fully Config-Driven
- No hardcoded parameters, paths, or intervals
- Easy to modify training settings without code changes
- YAML configuration with validation

### ✅ Modular Architecture
- Easy to add new agents and environments
- Factory pattern for agent/environment creation
- Pluggable components

### ✅ Automatic Checkpointing
- Configurable save intervals
- Resume from any checkpoint
- Best model saving
- Automatic cleanup of old checkpoints

### ✅ Comprehensive Plotting
- Training metrics (rewards, episode lengths, errors)
- State-action trajectory plots
- Rolling averages with configurable windows
- High-quality plots with customizable styling

### ✅ Training Statistics
- RecordEpisodeStatistics integration
- Comprehensive logging and metrics
- Performance analysis and summaries
- JSON and numpy file exports

### ✅ Unique Run Management
- Timestamped run directories
- Configuration snapshots
- Organized experiment tracking
- No overwrites of previous runs

## 🔧 Supported Agents

| Agent Type | Description | Configuration File |
|------------|-------------|-------------------|
| `ADHDP` | Adaptive Dynamic Programming | `configs/adhdp_config.yaml` |
| `q_learning` | Tabular Q-Learning | `configs/q_learning_config.yaml` |
| `dqn` | Deep Q-Network | `configs/dqn_config.yaml` |
| `td3` | Twin Delayed Deep Deterministic | Coming soon |
| `actor_critic` | Actor-Critic | Coming soon |

## 🌍 Supported Environments

| Environment | Description | Configuration |
|-------------|-------------|---------------|
| `f16` | F16 Linear Model | LinearModelF16 with configurable reference signals |
| `lunarlander` | Gymnasium LunarLander | Standard and continuous variants |
| `invertedpendulum` | Custom Inverted Pendulum | Configurable physics parameters |

## 📈 Output Structure

Each training run creates a unique directory with:

```
experiments/[agent_name]/run_[timestamp]/
├── config.yaml                           # Configuration snapshot
├── checkpoint_ep[N]/                     # Periodic checkpoints
│   ├── checkpoint_ep[N]_brain.npz       # Agent parameters
│   ├── returns.npy                      # Episode rewards
│   ├── lengths.npy                      # Episode lengths
│   └── training_error.npy               # Training errors
├── checkpoint_final_ep[N]/               # Final checkpoint
├── checkpoint_best/                      # Best performing checkpoint
├── training_metrics_ep[N].png            # Training plots
├── state_evolution_ep[N].png             # Trajectory plots
├── training_summary.json                 # Final statistics
└── results.txt                          # Human-readable summary
```

## 🔍 Advanced Usage

### Custom Reference Signals
Configure complex reference signals in the environment config:

```yaml
environment:
  reference_config:
    1:                        # State index
      type: "cos_step"        # Signal type
      amp_range: [-20, 20]    # Amplitude range (degrees)
      n_levels: 15            # Discrete levels
      T_step: 5.0             # Step duration
```

### Custom Plotting
Control plot generation with detailed settings:

```yaml
plotting:
  figure_size: [12, 8]        # Plot dimensions
  dpi: 150                    # Resolution
  style: "seaborn"            # Matplotlib style
  training_metrics:
    rolling_window: 100       # Moving average window
    save_individual: true     # Individual metric plots
```

### Performance Monitoring
Track and analyze training performance:

```yaml
evaluation:
  rolling_window: 100         # Performance evaluation window
  convergence_threshold: 0.05 # Convergence detection
  stability_episodes: 50      # Stability analysis
```

## 🛠️ Development

### Adding New Agents
1. Implement agent in `agents/[agent_name].py`
2. Add factory function in `utils/agent_factory.py`
3. Create configuration template in `configs/`

### Adding New Environments
1. Implement environment in `envs/[env_name].py`
2. Add factory function in `utils/environment_factory.py`
3. Update configuration templates

### Custom Metrics
Agents can provide custom metrics via `get_metrics()` method:

```python
def get_metrics(self):
    return {
        'custom_loss': self.loss_history[-1],
        'exploration_rate': self.epsilon
    }
```

## 📊 Monitoring Training

Monitor training progress through:
- Real-time console output with progress bars
- Periodic checkpoint saves
- Training metric plots
- JSON statistics files
- Human-readable results summaries

## 🏃‍♂️ Performance

The refactored system is designed for efficiency:
- Minimal overhead from modular architecture
- Efficient memory usage for trajectory storage
- Configurable logging to reduce I/O
- Optimized plotting with selective generation

## 🤝 Compatibility

- Fully compatible with existing agents and environments
- Gymnasium-based environment interface
- PyTorch and NumPy backend support
- Cross-platform (Windows, Linux, macOS)

---

## Migration Guide

If migrating from the old system:

1. **Configurations**: Convert hardcoded parameters to YAML configs
2. **Training Scripts**: Replace old scripts with `train.py --config [config.yaml]`
3. **Checkpoints**: Old checkpoints are compatible with the new system
4. **Plotting**: New system provides enhanced plotting capabilities

The refactored system maintains backward compatibility while providing significant improvements in modularity, configurability, and usability.