# Short-Period Pitch Dynamics Environment

A Gymnasium-compliant environment implementing linearized short-period pitch dynamics for reinforcement learning research.

## Overview

The `ShortPeriodEnv` simulates the short-period longitudinal dynamics of an aircraft around a trim condition. The agent must track a sinusoidal angle-of-attack reference signal by controlling the elevator deflection.

### State Representation

**State Vector**: `x = [α, q]ᵀ`
- `α`: Angle of attack [rad]
- `q`: Pitch rate [rad/s]

**Observation Vector**: `[α, q, α_ref]`
- Includes current state plus reference signal

### Dynamics

Linear state-space model:
```
ẋ = Ax + Bu + w(t)
y = Cx
```

**Default Matrices**:
```python
A = [[-0.5,  1.0],
     [-20.0, -2.0]]

B = [[0.0],
     [5.0]]

C = [[1.0, 0.0],
     [0.0, 1.0]]
```

### Reference Signal

Sinusoidal tracking:
```
α_ref(t) = A_ref × sin(2πt / T_ref)
```

Default: `A_ref = 0.1 rad` (~5.7°), `T_ref = 10.0 s`

## Configuration

All environment parameters are configured via YAML file: `configs/environments/shortperiod.yaml`

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.02 | Integration time step [s] |
| `max_steps` | 1000 | Maximum episode length |
| `reference.amplitude` | 0.1 | Reference amplitude [rad] |
| `reference.period` | 10.0 | Reference period [s] |
| `reward.w_alpha` | 100.0 | Tracking error weight |
| `reward.w_q` | 10.0 | Pitch rate penalty |
| `reward.w_u` | 1.0 | Control effort penalty |
| `noise.process_noise_std` | 0.0 | Process noise std dev |

### Example Configuration

```yaml
# configs/environments/shortperiod.yaml
name: "shortperiod"

dynamics:
  A: [[-0.5, 1.0], [-20.0, -2.0]]
  B: [[0.0], [5.0]]
  C: null  # Uses identity

dt: 0.02
max_steps: 1000

reference:
  amplitude: 0.1    # 0.1 rad ≈ 5.7°
  period: 10.0      # 10 second period

reward:
  w_alpha: 100.0    # Tracking error weight
  w_q: 10.0         # Pitch rate penalty
  w_u: 1.0          # Control effort penalty

noise:
  process_noise_std: 0.0  # No noise by default

training:
  episodes: 5000
  seed: 42
```

## Usage

### 1. Standalone Usage

```python
from envs.shortperiod_env import ShortPeriodEnv
import numpy as np

# Create environment
env = ShortPeriodEnv(
    A_ref=0.2,      # Larger amplitude
    T_ref=5.0,      # Faster reference
    dt=0.02
)

# Run episode
obs, info = env.reset(seed=42)
for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# Visualize
env.render()
```

### 2. With Training Framework

```bash
# Train with SAC agent
python train.py \
    --env-config configs/environments/shortperiod.yaml \
    --agent-config configs/agents/sac.yaml

# Train with DQN agent
python train.py \
    --env-config configs/environments/shortperiod.yaml \
    --agent-config configs/agents/dqn.yaml

# Train with IDHP agent
python train.py \
    --env-config configs/environments/shortperiod.yaml \
    --agent-config configs/agents/idhp.yaml
```

### 3. Resume Training

```bash
python train.py \
    --env-config configs/environments/shortperiod.yaml \
    --agent-config configs/agents/sac.yaml \
    --resume-from experiments/sac/run_20251015_123456/checkpoint_final_ep5000/checkpoint.pkl
```

## Reward Function

The reward at each step is:

```
r = -(w_α × (α - α_ref)² + w_q × q² + w_u × δ_e²)
```

- Penalizes tracking error, pitch rate deviation, and control effort
- If episode terminates early (constraint violation): `r += -1000`

## Termination Conditions

**Terminated** (constraint violation):
- `|α| > 0.4 rad` (~23°)
- `|q| > 200 rad/s`

**Truncated** (time limit):
- `steps ≥ max_episode_steps`

## Observation & Action Spaces

**Observation Space**:
```python
Box(low=[-0.35, -50.0, -0.35],
    high=[0.35, 50.0, 0.35],
    shape=(3,), dtype=float32)
```

**Action Space**:
```python
Box(low=-0.436,   # -25° in radians
    high=0.436,   # +25° in radians
    shape=(1,), dtype=float32)
```

## Customization Examples

### Modify System Dynamics

```yaml
# configs/environments/shortperiod.yaml
dynamics:
  # Faster dynamics
  A: [[-0.8, 1.2], [-25.0, -3.0]]
  # More control authority
  B: [[0.0], [8.0]]
```

### Change Reference Signal

```yaml
# Larger amplitude, faster tracking
reference:
  amplitude: 0.2    # ~11.5°
  period: 5.0       # 5 second period
```

### Add Process Noise

```yaml
# Add robustness challenge
noise:
  process_noise_std: 0.01  # Small Gaussian noise
```

### Adjust Reward Weights

```yaml
# Emphasize tracking accuracy
reward:
  w_alpha: 200.0    # Higher tracking weight
  w_q: 5.0          # Lower pitch rate penalty
  w_u: 0.5          # Lower control penalty
```

## Performance Metrics

The training framework automatically logs:

- **Episode reward**: Total reward per episode
- **Episode length**: Steps until termination/truncation
- **Tracking error**: RMS error between α and α_ref
- **Control effort**: RMS elevator deflection

Plots are saved in `experiments/{agent}/run_{timestamp}/plots/`

## Checkpoints

Checkpoints are saved at regular intervals in:
```
experiments/{agent}/run_{timestamp}/checkpoint_ep{N}/
```

Each checkpoint contains:
- `checkpoint.pkl`: Agent state, environment state, episode number
- `config.yaml`: Full configuration snapshot

## Visualization

After training, visualize results:

```python
# Plot trajectory from specific episode
from utils.plotting_manager import PlottingManager
import matplotlib.pyplot as plt

# Trajectories are auto-saved during training
# Check: experiments/{agent}/run_{timestamp}/plots/trajectory_ep{N}.png
```

The `render()` method shows:
1. Angle of attack (α) vs reference (α_ref)
2. Pitch rate (q)
3. Elevator deflection (δ_e)

## Tips for Good Performance

1. **Start with SAC or TD3** for continuous control
2. **Tune reward weights** to balance tracking vs control effort
3. **Adjust reference amplitude** based on achievable performance
4. **Monitor tracking error** in training plots
5. **Check constraint violations** - frequent terminations indicate:
   - Reward weights need adjustment
   - Dynamics too aggressive
   - Initial learning rate too high

## Integration Verification

Test the integration:

```python
import yaml
from utils.environment_factory import create_environment

# Load config
with open('configs/environments/shortperiod.yaml') as f:
    config = yaml.safe_load(f)

# Create environment
env = create_environment(config)

# Test
obs, _ = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

print(f"✓ Integration successful!")
print(f"  Observation: {obs}")
print(f"  Reward: {reward:.4f}")
```

## References

- Gymnasium: https://gymnasium.farama.org/
- Stevens & Lewis, "Aircraft Control and Simulation"
- Short-period approximation: longitudinal dynamics, fast oscillatory mode

---

**File locations**:
- Environment: `envs/shortperiod_env.py`
- Config: `configs/environments/shortperiod.yaml`
- Factory: `utils/environment_factory.py`
