from base_agent import Agent
import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim

# TODO: FINISH THIS CLASS
class DQN(Agent):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        obs_discretizer=None,
        action_discretizer=None,
        hidden_sizes: tuple = (64, 64),
        device: str = "cpu",
    ):
        super().__init__(
            env,
            learning_rate,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
            obs_discretizer,
            action_discretizer,
        )
        self.device = device
        input_dim = np.prod(env.observation_space.shape)
        output_dim = np.prod(env.action_space.shape)
        
        layers = []
        last_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.ReLU())
            last_size = h
        layers.append(nn.Linear(last_size, output_dim))
        self.q_network = nn.Sequential(*layers).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def get_action(self, obs: tuple) -> tuple:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        if np.random.rand(0 < self.epsilon):
            action = self.agent_action_space.space.sample()
        else:
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
            action_index = q_values.argmax().item()
            action = self.agent_action_space.undiscretize((action_index,))