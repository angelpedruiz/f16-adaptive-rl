from torch import nn
import torch
import numpy as np
import gimnasium as gym

class Actor(nn.Module):
    def __init__(self):
        '''Initialize the Actor network.'''
        pass

    def forward(self, state):
        '''Define the forward pass of the Actor network.'''
        pass
    
    def update(self):
        '''Define the update mechanism for the Actor network.'''
        pass
    
    def _exploration(self):
        '''Define the exploration strategy for the Actor network.'''
        pass  
    
    def _apply_alpha_decay(self):
        '''Apply alpha decay to the Actor network.'''
        pass
    
class Critic(nn.Module):
    def __init__(self):
        '''Initialize the Critic network.'''
        pass

    def forward(self, state, action):
        '''Define the forward pass of the Critic network.'''
        pass
    
    def update(self):
        '''Define the update mechanism for the Critic network.'''
        pass
    
class IncrementalModel():

    def __init__(self, obs_dim: int, act_dim: int, forgetting_factor: float = 0.99, delta: float = 1.0):
        pass
    
    def predict(self):
        pass

    def update(self, x: np.ndarray, u: np.ndarray, next_x: np.ndarray):
        pass


class IHDPAgent():
    def __init__(self, obs_space: gym.spaces.Box, act_space: gym.spaces.Box, parameters: dict):
        self.gamma = parameters.get('gamma')
        self.actor_lr = parameters.get('learning_rate')
        self.critic_lr = parameters.get('critic_learning_rate')
        

        self.actor = Actor()
        self.critic = Critic()
        self.model = IncrementalModel()
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        '''Get action from the agent, to return to environment.'''
        return self.actor(obs)

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, next_obs: np.ndarray):
        '''Update the agent's networks and models. main iteration loop: 1:agent.get_action(); 2:env.step(); 3:agent.update()'''
        xt = obs
        ut = action
        r = reward
        xt1 = next_obs
        
        
        # Predict next state
        xt1est = self.model.predict(xt, ut)
        G = self.model.G
        
        # Update critic
        Vt1, dVt1dxt1 = self.critic.update(xt, ut, r, xt1, terminated, xt1est)
        
        # Update actor
        self.actor.update(xt, Vt1, dVt1dxt1, G)