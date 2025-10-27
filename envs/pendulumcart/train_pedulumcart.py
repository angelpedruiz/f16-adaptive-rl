'''
Holds trainer class to train RL agents on the PendulumCart environment. Main entry point to train agents.
'''
import sys
import os
import numpy as np
import torch
import tqdm

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pendulumcart import PendulumCartEnv
from agents.HDP.hdp import HDPAgent
from agents.IHDP.ihdp import IHDPAgent
from utils.plotting_manager import PlottingManager

class PendulumCartTrainer():
    def __init__(self, env: PendulumCartEnv):
        self.env = env
        
    def train_hdp(self, agent: HDPAgent, max_steps: int) -> dict:
        """
        Train an HDP agent on the PendulumCart environment.

        Args:
            agent: HDPAgent instance to train
            max_steps: Maximum number of training steps

        Returns:
            Dictionary containing training data for visualization
        """

        # Initialize training data storage
        training_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'critic_errors': [],
            'critic_predictions': [],
            'critic_targets': [],
            'model_errors': [],
            'model_predictions': [],
            'true_states': [],
            'losses': {
                'actor': [],
                'critic': [],
                'model': []
            },
            'weight_norms': {
                'actor': [],
                'critic': [],
                'model': []
            },
            'weight_update_norms': {
                'actor': [],
                'critic': [],
                'model': []
            },
            'gradient_norms': {
                'actor': [],
                'critic': [],
                'model': []
            },
        }

        # Initialize environment
        obs, _ = self.env.reset()

        # Initialize agent memory
        agent.prev_obs = torch.FloatTensor(obs).unsqueeze(0)
        agent.prev_reward = torch.FloatTensor([0.0]).unsqueeze(0)

        for step in tqdm.tqdm(range(max_steps), desc="Training HDP Agent", unit="step"):
            # Get action from agent
            action = agent.get_action(obs)

            # Take step in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # PRINT OUTPUT OF EACH LAYER IN ACTOR NETWORK
            if step % 10 == 0:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

                # Forward pass through actor up to last linear layer (exclude tanh)
                x = obs_tensor
                for layer in agent.actor.model[:-1]:  # all layers except last tanh
                    x = layer(x)
                    if isinstance(layer, torch.nn.Linear):
                        print("Actor layer pre-activation mean/std:", x.mean().item(), x.std().item())


            # Update agent and get metrics
            metrics = agent.update(obs, action, reward, terminated, next_obs)
            
            # Store data every max_steps // 10
            training_data['states'].append(env.state.copy())
            training_data['actions'].append(action)
            training_data['rewards'].append(reward)
            training_data['losses']['actor'].append(metrics['losses']['actor_loss'])
            training_data['losses']['critic'].append(metrics['losses']['critic_loss'])
            training_data['losses']['model'].append(metrics['losses']['model_loss'])
            training_data['critic_errors'].append(metrics['critic_error'])
            training_data['critic_predictions'].append(metrics['critic_prediction'])
            training_data['critic_targets'].append(metrics['critic_target'])
            training_data['model_errors'].append(metrics['model_error'])
            training_data['model_predictions'].append(metrics['model_prediction'])
            training_data['true_states'].append(metrics['true_state'])
            for net in ['actor', 'critic', 'model']:
                training_data['weight_norms'][net].append(metrics['weights_norm'][net])
                training_data['weight_update_norms'][net].append(metrics['weights_update_norm'][net])
                training_data['gradient_norms'][net].append(metrics['gradients_norm'][net])

            # Handle episode end
            if done:
                obs, _ = self.env.reset()
                #break # only one episode
            else:
                obs = next_obs
                
        # print final actor parameters
        for name, param in agent.actor.state_dict().items():
            print(f"{name}: {param}")
            
        print("\n=== Actor Gradients ===")
        for name, param in agent.actor.named_parameters():
            if param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm().item()}")
            else:
                print(f"{name} grad is None")


        return training_data
    
    def train_ihdp(self, agent, max_steps: int) -> dict:
        """
        Train an IHDP (Incremental Heuristic Dynamic Programming) agent on the PendulumCart environment.

        IHDP uses a linear RLS model instead of a deep neural network, making it computationally efficient
        and allowing for adaptive control through explicit Jacobian (F, G) matrix extraction.

        Args:
            agent: IHDPAgent instance to train
            max_steps: Maximum number of training steps

        Returns:
            Dictionary containing training data for visualization
        """

        # Initialize training data storage
        training_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'critic_errors': [],
            'critic_predictions': [],
            'critic_targets': [],
            'model_errors': [],
            'model_predictions': [],
            'true_states': [],
            'losses': {
                'actor': [],
                'critic': [],
            },
            'weight_norms': {
                'actor': [],
                'critic': [],
            },
            'weight_update_norms': {
                'actor': [],
                'critic': [],
            },
            'gradient_norms': {
                'actor': [],
                'critic': [],
            },
        }

        # Initialize environment
        obs, _ = self.env.reset()

        # Initialize agent memory
        agent.prev_obs = torch.FloatTensor(obs).unsqueeze(0)
        agent.prev_reward = torch.FloatTensor([0.0]).unsqueeze(0)

        for step in tqdm.tqdm(range(max_steps), desc="Training IHDP Agent", unit="step"):
            # Get action from agent
            action = agent.get_action(obs)

            # Take step in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Update agent and get metrics
            metrics = agent.update(obs, action, reward, terminated, next_obs)

            # Store data
            training_data['states'].append(self.env.state.copy())
            training_data['actions'].append(action)
            training_data['rewards'].append(reward)
            training_data['losses']['actor'].append(metrics['losses']['actor_loss'])
            training_data['losses']['critic'].append(metrics['losses']['critic_loss'])
            training_data['critic_errors'].append(metrics['critic_error'])
            training_data['critic_predictions'].append(metrics['critic_prediction'])
            training_data['critic_targets'].append(metrics['critic_target'])
            training_data['model_errors'].append(metrics['model_error'])
            training_data['model_predictions'].append(metrics['model_prediction'])
            training_data['true_states'].append(metrics['true_state'])
            for net in ['actor', 'critic']:
                training_data['weight_norms'][net].append(metrics['weights_norm'][net])
                training_data['weight_update_norms'][net].append(metrics['weights_update_norm'][net])
                training_data['gradient_norms'][net].append(metrics['gradients_norm'][net])

            # Handle episode end
            if done:
                obs, _ = self.env.reset()
                # Reinitialize agent memory for new episode
                agent.prev_obs = torch.FloatTensor(obs).unsqueeze(0)
                agent.prev_reward = torch.FloatTensor([0.0]).unsqueeze(0)
            else:
                obs = next_obs

        # Print final actor parameters
        for name, param in agent.actor.state_dict().items():
            print(f"{name}: {param}")

        print("\n=== Actor Gradients ===")
        for name, param in agent.actor.named_parameters():
            if param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm().item()}")
            else:
                print(f"{name} grad is None")

        # Print final RLS model Jacobians
        F, G = agent.model.get_jacobians()
        print("\n=== RLS Model Jacobians ===")
        print(f"F (state transition) shape: {F.shape}")
        print(f"F:\n{F}")
        print(f"\nG (control sensitivity) shape: {G.shape}")
        print(f"G:\n{G}")

        return training_data
    
if __name__ == "__main__":
    from datetime import datetime
    
    # seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------- Parameters -------
    max_steps_per_episode = 300
    training_max_steps = 1000
    dt = 0.01

    forgetting_factor = 0.8
    initial_covariance = 0.99
    gamma = 0.99
    lr_actor = 5e-2
    lr_critic = 1e-1
    lr_model = 5e-1
    actor_sizes = [6, 6]
    critic_sizes = [6, 6]
    model_sizes = [10, 10]

    # Create timestamped results directory with hierarchical structure
    # Structure: results/[env_name]/[agent_name]/[timestamp]/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = 'pendulumcart'
    agent_name = 'IHDP'

    # Create directory for IHDP agent
    save_dir = os.path.join('./results', env_name, agent_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}\n")

    # ------- Initialize -------
    env = PendulumCartEnv(dt=dt, max_steps=max_steps_per_episode)
    trainer = PendulumCartTrainer(env)
    agentHDP = HDPAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        gamma=gamma,
        hidden_sizes={'actor': actor_sizes, 'critic': critic_sizes, 'model': model_sizes},
        learning_rates={'actor': lr_actor, 'critic': lr_critic, 'model': lr_model}
    )
    agent_ihdp = IHDPAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        gamma=gamma,
        forgetting_factor=0.99,
        initial_covariance=1.0,
        hidden_sizes={'actor': actor_sizes, 'critic': critic_sizes,},
        learning_rates={'actor': lr_actor, 'critic': lr_critic}
    )

    # ------- Train Agent -------
    print("Starting training...")
    training_data = trainer.train_ihdp(agent_ihdp, max_steps=training_max_steps)
    print("Training complete!")
    print("Plotting results...")

    # ------- Convert data to numpy arrays for plotting -------
    training_data['states'] = np.array(training_data['states'])
    training_data['actions'] = np.array(training_data['actions'])
    training_data['rewards'] = np.array(training_data['rewards'])

    # ------- Plotting -------
    plotting_manager = PlottingManager(env=env, agent=agent_ihdp, save_dir=save_dir)

    # Render animation of the episode
    plotting_manager.render_pendulumcart_env(
        states=training_data['states'],
        filename='pendulum_animation.gif',
        fps=30
    )

    # Plot trajectory
    plotting_manager.plot_pendulumcart_trajectory(
        states=training_data['states'],
        actions=training_data['actions'],
        rewards=training_data['rewards']
    )

    # Plot learning metrics
    plotting_manager.plot_ihdp_learning(training_data)

    # Save env and agent parameters
    plotting_manager.save_run_params(seed=seed) # last snapshot

    print("\nAll plots and animations generated successfully!")
    print(f"Results saved to: {save_dir}")
    
    
    
    
    
    