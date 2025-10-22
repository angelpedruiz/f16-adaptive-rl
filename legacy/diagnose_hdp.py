"""
Diagnose why HDP actor produces small actions.
"""
import numpy as np
import torch
from envs.pendulumcart import PendulumCartEnv
from agents.hdp import HDPAgent

def diagnose_actor():
    """Check actor network outputs and gradients."""

    env = PendulumCartEnv()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create agent
    agent = HDPAgent(
        obs_dim=env.state_dim,
        act_dim=env.act_dim,
        hidden_sizes=[64, 32],
        actor_lr=1e-4,
        critic_lr=5e-3,
        model_lr=1e-2,
        gamma=0.95,
        device=device,
        action_low=np.array([-env.max_force]),
        action_high=np.array([env.max_force])
    )

    print("=" * 60)
    print("DIAGNOSING HDP ACTOR")
    print("=" * 60)

    # Check initial actor output
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    print("\n1. INITIAL ACTOR OUTPUT (before training)")
    print("-" * 60)
    with torch.no_grad():
        actor_output_raw = agent.actor(obs_tensor)
        print(f"   Raw actor output (tanh): {actor_output_raw.cpu().numpy().flatten()}")

    action = agent.get_action(obs)
    print(f"   Scaled action:           {action}")
    print(f"   Action bounds:           [{-env.max_force}, {env.max_force}]")
    print(f"   Action as % of max:      {abs(action[0])/env.max_force*100:.2f}%")

    # Check actor network parameters
    print("\n2. ACTOR NETWORK WEIGHTS")
    print("-" * 60)
    for name, param in agent.actor.named_parameters():
        if 'weight' in name:
            print(f"   {name:30s} mean: {param.mean().item():.6f}, std: {param.std().item():.6f}")
            print(f"   {'':<30s} min:  {param.min().item():.6f}, max: {param.max().item():.6f}")

    # Train for a few steps and check gradients
    print("\n3. GRADIENT FLOW AFTER FIRST UPDATE")
    print("-" * 60)

    obs, _ = env.reset()
    action = agent.get_action(obs)
    next_obs, reward, terminated, _, _ = env.step(action)

    # First update (will return zeros)
    losses = agent.update(obs, action, reward, terminated, next_obs)
    print(f"   First update losses: {losses}")

    # Second update (actual training)
    obs = next_obs
    action = agent.get_action(obs)
    next_obs, reward, terminated, _, _ = env.step(action)
    losses = agent.update(obs, action, reward, terminated, next_obs)

    print(f"   Second update losses:")
    print(f"      Actor loss:  {losses['actor_loss']:.6f}")
    print(f"      Critic loss: {losses['critic_loss']:.6f}")
    print(f"      Model loss:  {losses['model_loss']:.6f}")

    # Check gradients
    print("\n4. ACTOR GRADIENTS (after second update)")
    print("-" * 60)
    for name, param in agent.actor.named_parameters():
        if param.grad is not None and 'weight' in name:
            grad_norm = param.grad.norm().item()
            print(f"   {name:30s} grad norm: {grad_norm:.8f}")

    # Check value estimates
    print("\n5. VALUE ESTIMATES")
    print("-" * 60)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        V_current = agent.critic(obs_tensor)
        print(f"   V(current state): {V_current.item():.6f}")

        actor_action = agent.actor(obs_tensor)
        model_input = torch.cat([obs_tensor, actor_action], dim=-1)
        next_state_pred = agent.model(model_input)
        V_next_pred = agent.critic(next_state_pred)
        print(f"   V(predicted next state): {V_next_pred.item():.6f}")
        print(f"   Actor loss would be: {-V_next_pred.item():.6f}")

    # Train for 10 more episodes and check progress
    print("\n6. TRAINING FOR 10 EPISODES")
    print("-" * 60)

    for ep in range(10):
        obs, _ = env.reset()
        ep_reward = 0
        ep_actions = []

        for step in range(50):
            action = agent.get_action(obs)
            ep_actions.append(action[0])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            losses = agent.update(obs, action, reward, terminated, next_obs)
            ep_reward += reward
            obs = next_obs

            if terminated or truncated:
                break

        print(f"   Episode {ep+1:2d}: Reward={ep_reward:6.2f}, "
              f"Actions: mean={np.mean(ep_actions):6.3f}, "
              f"std={np.std(ep_actions):6.3f}, "
              f"max={np.max(np.abs(ep_actions)):6.3f}")

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_actor()
