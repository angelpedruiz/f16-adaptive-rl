import sys
import os
import numpy as np
import torch
import tqdm
from datetime import datetime

from agents.IHDP.ihdp import IHDPAgent
from envs.shortperiod.shortperiod import ShortPeriodEnv
from utils.plotting_manager import PlottingManager

def train_ihdp(env: ShortPeriodEnv, agent: IHDPAgent, max_steps: int) -> dict:
    training_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "critic_errors": [],
            "critic_predictions": [],
            "critic_targets": [],
            "angles": [],  # Pendulum angle (theta)
            "neg_abs_angles": [],  # -abs(theta) as reference signal
            "model_errors": [],
            "model_predictions": [],
            "true_states": [],
            "losses": {
                "actor": [],
                "critic": [],
            },
            "weight_norms": {
                "actor": [],
                "critic": [],
            },
            "weight_update_norms": {
                "actor": [],
                "critic": [],
            },
            "gradient_norms": {
                "actor": [],
                "critic": [],
            },
            "actor_weights_history": [],
            "critic_weights_history": [],
            "dVdx_history": [],
            "alpha_refs": [],
        }

    obs, _ = env.reset()
    
    for step in tqdm.tqdm(range(max_steps), desc="Training IHDP Agent", unit="step"):
        action = agent.get_action(obs) # [-1,1] scaled action
        scaled_action = env.action_space.low + (0.5 * (action + 1.0) * (env.action_space.high - env.action_space.low))
        action = scaled_action.astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        metrics = agent.update(obs, action, reward, terminated, next_obs)

        # Logging
        training_data["states"].append(env.state.copy())
        training_data["actions"].append(info['delta_e'])
        training_data["rewards"].append(reward)
        training_data["alpha_refs"].append(info["alpha_ref"])
        training_data["losses"]["actor"].append(metrics["losses"]["actor_loss"])
        training_data["losses"]["critic"].append(metrics["losses"]["critic_loss"])
        training_data["critic_errors"].append(metrics["critic_error"])
        training_data["critic_predictions"].append(metrics["critic_prediction"])
        training_data["critic_targets"].append(metrics["critic_target"])
        training_data["angles"].append(metrics["theta"])
        training_data["neg_abs_angles"].append(-abs(metrics["theta"]))
        training_data["model_errors"].append(metrics["model_error"])
        training_data["model_predictions"].append(metrics["model_prediction"])
        training_data["true_states"].append(metrics["true_state"])
        for net in ["actor", "critic"]:
            training_data["weight_norms"][net].append(metrics["weights_norm"][net])
            training_data["weight_update_norms"][net].append(
                metrics["weights_update_norm"][net]
            )
            training_data["gradient_norms"][net].append(
                metrics["gradients_norm"][net]
            )

        # Store weight history from metrics
        training_data["actor_weights_history"].append(metrics["actor_weights"])
        training_data["critic_weights_history"].append(metrics["critic_weights"])
        training_data["dVdx_history"].append(metrics["dVdx"])
        
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
    return training_data

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

max_steps_per_episode = 600
training_max_steps = 1200
dt = 0.01

forgetting_factor = 0.8
initial_covariance = 0.99
gamma = 0.0
lr_actor = 0.01
lr_critic = 1e-2
actor_sizes = [6, 6]
critic_sizes = [6, 6]
model_sizes = [10, 10]
actor_weight_limit = 0.5
critic_weight_limit =0.5

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
env_name = "shortperiod"
agent_name = "IHDP"
save_dir = os.path.join("./results", env_name, agent_name, timestamp)
os.makedirs(save_dir, exist_ok=True)
print(f"Results will be saved to: {save_dir}\n")

env = ShortPeriodEnv(dt=dt, max_steps=max_steps_per_episode)
agent_ihdp = IHDPAgent(
    obs_space=env.obs_space,
    act_space=env.action_space,
    gamma=gamma,
    forgetting_factor=forgetting_factor,
    initial_covariance=initial_covariance,
    hidden_sizes={"actor": actor_sizes, "critic": critic_sizes},
    learning_rates={"actor": lr_actor, "critic": lr_critic},
    actor_weight_limit=actor_weight_limit,
    critic_weight_limit=critic_weight_limit,
)
print("Starting training...")
training_data = train_ihdp(
    env=env,
    agent=agent_ihdp,
    max_steps=training_max_steps,
)
print("Training complete!")
print("Plotting results...")

# ------- Convert data to numpy arrays for plotting -------
training_data["states"] = np.array(training_data["states"])
training_data["actions"] = np.array(training_data["actions"]).reshape(-1, 1)  # Reshape to (timesteps, 1)
training_data["rewards"] = np.array(training_data["rewards"])
training_data["alpha_refs"] = np.array(training_data["alpha_refs"])

# ------- Plotting -------
plotting_manager = PlottingManager(env=env, agent=agent_ihdp, save_dir=save_dir)

plotting_manager.plot_shortperiod_trajectory(
    states=training_data["states"],
    actions=training_data["actions"],
    alpha_refs=training_data["alpha_refs"],
)

plotting_manager.plot_ihdp_learning(training_data, skip_critic_vs_angle=True)
plotting_manager.save_run_params(seed=seed)
print("\nAll plots and animations generated successfully!")
print(f"Results saved to: {save_dir}")