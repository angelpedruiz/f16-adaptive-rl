'''
Holds trainer class to train RL agents on the PendulumCart environment. Main entry point to train agents.
'''
import sys
import os
import numpy as np
import torch

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pendulumcart import PendulumCartEnv
from agents.HDP.hdp import HDPAgent
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

        for _ in range(max_steps):
            # Get action from agent
            action = agent.get_action(obs)

            # Take step in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Update agent and get metrics
            metrics = agent.update(obs, action, reward, terminated, next_obs)
            
            # Store data
            training_data['states'].append(env.state.copy())
            training_data['actions'].append(action)
            training_data['rewards'].append(reward)
            training_data['losses']['actor'].append(metrics['losses']['actor_loss'])
            training_data['losses']['critic'].append(metrics['losses']['critic_loss'])
            training_data['losses']['model'].append(metrics['losses']['model_loss'])
            training_data['critic_errors'].append(metrics['critic_error'])
            training_data['model_errors'].append(metrics['model_error'])
            training_data['model_predictions'].append(metrics['model_prediction'])
            training_data['true_states'].append(metrics['true_state'])
            for net in ['actor', 'critic', 'model']:
                training_data['weight_norms'][net].append(metrics['weights_norm'][net])
                training_data['weight_update_norms'][net].append(metrics['weights_update_norm'][net])
                training_data['gradient_norms'][net].append(metrics['gradients_norm'][net])

            # Handle episode end
            if done:
                break
            else:
                obs = next_obs

        return training_data
    
if __name__ == "__main__":
    from datetime import datetime

    # ------- Parameters -------
    max_steps = 100
    dt = 0.01

    gamma = 0.99
    lr_actor = 1e-5
    lr_critic = 1e-7
    lr_model = 1e-10
    actor_sizes = [32, 32]
    critic_sizes = [32, 32]
    model_sizes = [32, 32]

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('./results', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}\n")

    # ------- Initialize -------
    env = PendulumCartEnv(dt=dt, max_steps=max_steps)
    trainer = PendulumCartTrainer(env)
    agent = HDPAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        gamma=gamma,
        hidden_sizes={'actor': actor_sizes, 'critic': critic_sizes, 'model': model_sizes},
        learning_rates={'actor': lr_actor, 'critic': lr_critic, 'model': lr_model}
    )

    # ------- Train Agent -------
    print("Starting training...")
    training_data = trainer.train_hdp(agent, max_steps=max_steps)
    print("Training complete!")

    # ------- Convert data to numpy arrays for plotting -------
    training_data['states'] = np.array(training_data['states'])
    training_data['actions'] = np.array(training_data['actions'])
    training_data['rewards'] = np.array(training_data['rewards'])

    # ------- Plotting -------
    plotting_manager = PlottingManager(env=env, agent=agent, save_dir=save_dir)
    
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
    plotting_manager.plot_hdp_learning(training_data)

    print("\nAll plots and animations generated successfully!")
    print(f"Results saved to: {save_dir}")
    
    
    
    
    
    