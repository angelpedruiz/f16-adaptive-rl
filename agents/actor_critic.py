from agents.base_agent import Agent
from collections import defaultdict
import gymnasium as gym
import numpy as np


class ActorCritic(Agent):
    def __init__(
        self,
        env: gym.Env,
        discount_factor: float,
        temperature_decay: float,
        final_temperature: float,
        actor_lr: float,
        critic_lr: float,
        temperature: float = 1.0,
        state_discretizer=None,
        action_discretizer=None,
    ):
        super().__init__(
            env,
            learning_rate=0.0,  # No need for epsilon and learning rate in ActorCritic, set to 0 or adjust accordingly.
            initial_epsilon=0.0,
            epsilon_decay=0.0,
            final_epsilon=0.0,
            discount_factor=discount_factor,
            state_discretizer=state_discretizer,
            action_discretizer=action_discretizer,
        )

        # Additional Actor-Critic specific attributes
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.final_temperature = final_temperature

        self.actor_prefs = defaultdict(lambda: np.zeros(self.num_actions))
        self.critic_values = defaultdict(lambda: 0.0)
        self.training_error = []

    def softmax(self, preferences: np.ndarray) -> np.ndarray:
        scaled_prefs = preferences / self.temperature
        exp_preferences = np.exp(scaled_prefs - np.max(scaled_prefs))  # stability
        return exp_preferences / np.sum(exp_preferences)

    def get_action(self, obs: tuple) -> tuple:
        discretized_obs = self.agent_obs_space.discretize(obs)
        action_probs = self.softmax(self.actor_prefs[discretized_obs])
        action_index = np.random.choice(len(action_probs), p=action_probs)
        undiscretized_action = self.agent_action_space.undiscretize((action_index,))
        return undiscretized_action

    def decay_temperature(self):
        self.temperature = max(
            self.final_temperature, self.temperature * self.temperature_decay
        )

    def update(
        self,
        obs: tuple,
        action: tuple,
        reward: float,
        terminated: bool,
        next_obs: tuple,
    ):
        obs = self.agent_obs_space.discretize(obs)
        next_obs = self.agent_obs_space.discretize(next_obs)
        action_index = self.agent_action_space.discretize(action)[
            0
        ]  # Get the index of the action
        td_error = (
            reward
            + self.discount_factor * (not terminated) * self.critic_values[next_obs]
            - self.critic_values[obs]
        )
        self.critic_values[obs] += self.critic_lr * td_error

        # print(f'actions: {self.actor_prefs[obs]}')

        softmax_probs = self.softmax(self.actor_prefs[obs])
        for a in range(len(self.actor_prefs[obs])):
            if a == action_index:
                self.actor_prefs[obs][a] += (
                    self.actor_lr * td_error * (1 - softmax_probs[a])
                )
            else:
                self.actor_prefs[obs][a] -= self.actor_lr * td_error * softmax_probs[a]
        self.training_error.append(td_error)
