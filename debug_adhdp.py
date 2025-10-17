"""
Debug script for ADHDP learning on ShortPeriod environment.
Prints detailed diagnostics to identify learning issues.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from envs.shortperiod_env import ShortPeriodEnv
from agents.adhdp import ADHDPAgent


def main():
    # Create environment (match shortperiod_env.py settings)
    env = ShortPeriodEnv(
        dt=0.01,
        A_ref=0.35,
        T_ref=5.0,
        w_alpha=1.0,  # Key: smaller reward scale!
        w_q=0.0,
        w_u=0.0,
        max_episode_steps=1000
    )

    # Create agent (match shortperiod_env.py settings)
    gamma_val = 0.8
    actor_lr_val = 0.01
    critic_lr_val = 0.01
    agent = ADHDPAgent(
        obs_dim=3,
        act_dim=1,
        hidden_sizes=[32, 32],  # Smaller network trains faster
        actor_lr=actor_lr_val,  # Higher to prevent saturation
        critic_lr=critic_lr_val,  # Standard rate
        gamma=gamma_val,
        device='cuda',
        action_low=env.action_space.low,
        action_high=env.action_space.high
    )

    print("=" * 80)
    print("Testing online ADHDP (matching shortperiod_env.py settings):")
    print(f"  Learning rates: actor={actor_lr_val}, critic={critic_lr_val}")
    print("  Network size: [32, 32]")
    print("  Actor: Linear output (no tanh saturation)")
    print("  Reward scale: w_alpha=1.0 (10x smaller than before!)")
    print("  Episode length: 1000 steps")
    print(f"  Gamma: {gamma_val}")
    print("=" * 80)
    print("ADHDP Learning Diagnostics on ShortPeriod Environment")
    print("=" * 80)
    print("\nRunning single episode with detailed step-by-step diagnostics...")
    print("Looking for:")
    print("  1. TD error magnitude and whether it decreases over time")
    print("  2. Critic loss behavior (should decrease if learning)")
    print("  3. Actor loss sign and magnitude")
    print("  4. Value estimates (V_pred, V_next) coherence with rewards")
    print("  5. Action magnitude and changes")
    print("=" * 80)

    # Run single episode with detailed diagnostics
    obs, _ = env.reset(seed=42)
    total_reward = 0
    steps = 0
    terminated = False
    truncated = False

    # Track statistics
    debug_history = {
        'td_errors': [],
        'td_targets': [],
        'critic_losses': [],
        'actor_losses': [],
        'V_preds': [],
        'V_nexts': [],
        'rewards': [],
        'actions': [],
        'critic_grad_norm_pre': [],
        'critic_grad_norm_post': [],
        'actor_grad_norm_pre': [],
        'actor_grad_norm_post': [],
    }

    print("\nFirst 10 steps (detailed):")
    print("-" * 80)

    while not (terminated or truncated) and steps < 1000:
        # Get action
        action = agent.get_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Update agent with debugging
        debug_info = agent.update(obs, action, reward, terminated, next_obs, debug=True)

        # Track statistics
        debug_history['td_errors'].append(debug_info['td_error'])
        debug_history['td_targets'].append(debug_info['td_target'])
        debug_history['critic_losses'].append(debug_info['critic_loss'])
        debug_history['actor_losses'].append(debug_info['actor_loss'])
        debug_history['V_preds'].append(debug_info['V_pred'])
        debug_history['V_nexts'].append(debug_info['V_next'])
        debug_history['rewards'].append(debug_info['reward'])
        debug_history['actions'].append(action[0])
        debug_history['critic_grad_norm_pre'].append(debug_info['critic_grad_norm_pre'])
        debug_history['critic_grad_norm_post'].append(debug_info['critic_grad_norm_post'])
        debug_history['actor_grad_norm_pre'].append(debug_info['actor_grad_norm_pre'])
        debug_history['actor_grad_norm_post'].append(debug_info['actor_grad_norm_post'])

        # Print detailed info for first 10 steps
        if steps < 10:
            print(f"\nStep {steps}:")
            print(f"  State: alpha={np.rad2deg(obs[0]):6.2f}deg, q={obs[1]:6.3f} rad/s, alpha_ref={np.rad2deg(obs[2]):6.2f}deg")
            print(f"  Action: delta_e={action[0]:6.2f}deg (tanh_out={debug_info['current_action_magnitude']:6.3f})")
            print(f"  Reward: {debug_info['reward']:8.3f}")
            print(f"  Values: V(s,a)={debug_info['V_pred']:8.3f}, V(s',a')={debug_info['V_next']:8.3f}")
            print(f"  TD_target: {debug_info['td_target']:8.3f}, TD_error: {debug_info['td_error']:8.3f}")
            print(f"  Losses: Critic={debug_info['critic_loss']:8.3f}, Actor={debug_info['actor_loss']:8.3f}")
            print(f"  Gradients: Critic={debug_info['critic_grad_norm_pre']:6.3f}->{debug_info['critic_grad_norm_post']:6.3f}, Actor={debug_info['actor_grad_norm_pre']:6.3f}->{debug_info['actor_grad_norm_post']:6.3f}")

        total_reward += reward
        steps += 1
        obs = next_obs

    print("\n" + "=" * 80)
    print("EPISODE SUMMARY")
    print("=" * 80)

    # Convert to numpy arrays for statistics
    td_errors = np.array(debug_history['td_errors'])
    td_targets = np.array(debug_history['td_targets'])
    critic_losses = np.array(debug_history['critic_losses'])
    actor_losses = np.array(debug_history['actor_losses'])
    V_preds = np.array(debug_history['V_preds'])
    V_nexts = np.array(debug_history['V_nexts'])
    rewards = np.array(debug_history['rewards'])
    actions = np.array(debug_history['actions'])
    critic_grad_pre = np.array(debug_history['critic_grad_norm_pre'])
    critic_grad_post = np.array(debug_history['critic_grad_norm_post'])
    actor_grad_pre = np.array(debug_history['actor_grad_norm_pre'])
    actor_grad_post = np.array(debug_history['actor_grad_norm_post'])

    status = "TERMINATED" if terminated else ("TRUNCATED" if truncated else "ONGOING")
    print(f"\nStatus: {status} after {steps} steps")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final State: alpha={np.rad2deg(info['alpha']):.2f}deg, q={info['q']:.3f} rad/s")

    print("\n" + "-" * 80)
    print("STEP-BY-STEP LEARNING DYNAMICS")
    print("-" * 80)

    # Analyze TD error fluctuation around zero
    print("TD Error per-step behavior:")
    print(f"  Mean: {td_errors.mean():8.3f} (should oscillate around 0)")
    print(f"  Std:  {td_errors.std():8.3f}")
    print(f"  Sign changes: {np.sum(np.diff(np.sign(td_errors)) != 0)} / {len(td_errors)-1} steps")
    td_positive = np.sum(td_errors > 0)
    td_negative = np.sum(td_errors < 0)
    print(f"  Positive: {td_positive} ({100*td_positive/len(td_errors):.1f}%), Negative: {td_negative} ({100*td_negative/len(td_errors):.1f}%)")

    # Check if V_pred tracks td_target
    print("\nV_pred vs TD_target divergence:")
    divergence = np.abs(V_preds - td_targets)
    print(f"  Mean divergence: {divergence.mean():8.3f}")
    print(f"  First 10 steps:  {divergence[:10].mean():8.3f}")
    print(f"  Last  10 steps:  {divergence[-10:].mean():8.3f}")
    if divergence[-10:].mean() < divergence[:10].mean():
        print("  [OK] V_pred is converging to td_target")
    else:
        print("  [WARN] V_pred is NOT converging (may be diverging)")

    print("\n" + "-" * 80)
    print("GRADIENT ANALYSIS (Are networks actually learning?)")
    print("-" * 80)

    print("Critic gradients:")
    print(f"  Pre-clip  mean: {critic_grad_pre.mean():8.3f}, max: {critic_grad_pre.max():8.3f}")
    print(f"  Post-clip mean: {critic_grad_post.mean():8.3f}, max: {critic_grad_post.max():8.3f}")
    clipped_ratio = np.sum(critic_grad_pre > 1.0) / len(critic_grad_pre)
    print(f"  Clipped: {clipped_ratio*100:.1f}% of updates")
    if critic_grad_pre.mean() < 0.001:
        print("  [WARN] Critic gradients very small - may have vanished!")
    elif clipped_ratio > 0.5:
        print("  [WARN] Critic gradients clipped >50% of time - increase clip threshold!")
    else:
        print("  [OK] Critic gradients look healthy")

    print("\nActor gradients:")
    print(f"  Pre-clip  mean: {actor_grad_pre.mean():8.3f}, max: {actor_grad_pre.max():8.3f}")
    print(f"  Post-clip mean: {actor_grad_post.mean():8.3f}, max: {actor_grad_post.max():8.3f}")
    clipped_ratio_actor = np.sum(actor_grad_pre > 1.0) / len(actor_grad_pre)
    print(f"  Clipped: {clipped_ratio_actor*100:.1f}% of updates")

    # Check if actor is moving
    action_changes = np.abs(np.diff(actions))
    print("\nActor movement (action changes per step):")
    print(f"  Mean change: {action_changes.mean():6.3f} deg")
    print(f"  First 10 steps: {action_changes[:10].mean():6.3f} deg")
    print(f"  Last  10 steps: {action_changes[-10:].mean():6.3f} deg")
    if action_changes[-10:].mean() < 0.01:
        print("  [WARN] Actor stuck (actions not changing)")
    else:
        print("  [OK] Actor is moving")

    print("\n" + "-" * 80)
    print("CRITIC LEARNING ANALYSIS")
    print("-" * 80)

    # Analyze TD error trajectory
    first_100_td = np.abs(td_errors[:min(100, len(td_errors))])
    last_100_td = np.abs(td_errors[-min(100, len(td_errors)):])

    print("TD Error |magnitude| (not a reliable metric for online learning!):")
    print(f"  First steps: mean={first_100_td.mean():8.3f}, std={first_100_td.std():8.3f}")
    print(f"  Last  steps: mean={last_100_td.mean():8.3f}, std={last_100_td.std():8.3f}")

    print("\nCritic Loss:")
    print(f"  First steps: mean={critic_losses[:min(100, len(critic_losses))].mean():8.3f}")
    print(f"  Last  steps: mean={critic_losses[-min(100, len(critic_losses)):].mean():8.3f}")

    print("\n" + "-" * 80)
    print("ACTOR LEARNING ANALYSIS")
    print("-" * 80)

    print(f"Actor Loss:")
    print(f"  First 100 steps: mean={actor_losses[:100].mean():8.3f}")
    print(f"  Last  100 steps: mean={actor_losses[-100:].mean():8.3f}")
    print(f"  Overall:         mean={actor_losses.mean():8.3f}")
    print(f"  Sign: {'+' if actor_losses.mean() > 0 else '-'} (should be negative since we minimize -V)")

    print(f"\nAction Statistics:")
    print(f"  Mean: {actions.mean():6.2f}°, Std: {actions.std():6.2f}°")
    print(f"  Range: [{actions.min():6.2f}°, {actions.max():6.2f}°]")
    print(f"  First 10 actions: {actions[:10]}")
    print(f"  Last  10 actions: {actions[-10:]}")

    print("\n" + "-" * 80)
    print("VALUE FUNCTION ANALYSIS")
    print("-" * 80)

    print(f"V(s,a) predictions:")
    print(f"  First 100 steps: mean={V_preds[:100].mean():8.3f}, std={V_preds[:100].std():8.3f}")
    print(f"  Last  100 steps: mean={V_preds[-100:].mean():8.3f}, std={V_preds[-100:].std():8.3f}")
    print(f"  Range: [{V_preds.min():8.3f}, {V_preds.max():8.3f}]")

    print(f"\nV(s',a') predictions:")
    print(f"  First 100 steps: mean={V_nexts[:100].mean():8.3f}, std={V_nexts[:100].std():8.3f}")
    print(f"  Last  100 steps: mean={V_nexts[-100:].mean():8.3f}, std={V_nexts[-100:].std():8.3f}")
    print(f"  Range: [{V_nexts.min():8.3f}, {V_nexts.max():8.3f}]")

    print(f"\nReward Statistics:")
    print(f"  Mean: {rewards.mean():8.3f}, Std: {rewards.std():8.3f}")
    print(f"  Range: [{rewards.min():8.3f}, {rewards.max():8.3f}]")

    print("\n" + "-" * 80)
    print("COHERENCE CHECK")
    print("-" * 80)

    # Check if value estimates are in reasonable range compared to rewards
    print(f"Rewards are in range [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"V(s,a) should be in range [r_min/(1-gamma), r_max/(1-gamma)] ~ [{rewards.min()/(1-gamma_val):.3f}, {rewards.max()/(1-gamma_val):.3f}]")
    print(f"V(s,a) is actually in range [{V_preds.min():.3f}, {V_preds.max():.3f}]")

    expected_min = rewards.min()/(1-gamma_val)
    expected_max = rewards.max()/(1-gamma_val)
    if V_preds.min() >= expected_min * 0.1 and V_preds.max() <= expected_max * 2:
        print("  OK: Value estimates are in reasonable range")
    else:
        print("  WARNING: Value estimates may be out of expected range")
        print(f"    Expected: [{expected_min * 0.1:.1f}, {expected_max * 2:.1f}] (with tolerance)")
        print(f"    Actual:   [{V_preds.min():.1f}, {V_preds.max():.1f}]")

    # Check if TD error correlates with poor tracking
    tracking_errors = np.array([np.abs(debug_history['rewards'][i]) for i in range(len(debug_history['rewards']))])
    correlation = np.corrcoef(np.abs(td_errors), tracking_errors)[0, 1]
    print(f"\nCorrelation between |TD error| and |reward|: {correlation:.3f}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
