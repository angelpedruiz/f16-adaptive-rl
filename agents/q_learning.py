from agents.base_agent import Agent
import numpy as np


class QLearning(Agent):
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
        action = self.agent_action_space.discretize(action)
        assert all(
            0 <= a < n for a, n in zip(action, self.agent_action_space.space.nvec)
        ), (
            f"Discretized action {action} out of bounds for nvec {self.agent_action_space.space.nvec}"
        )
        flat_action = np.ravel_multi_index(action, self.agent_action_space.space.nvec)

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        td_error = (
            reward
            + self.discount_factor * future_q_value
            - self.q_values[obs][flat_action]
        )

        self.q_values[obs][flat_action] += self.lr * td_error
        self.training_error.append(td_error)
        
    def get_brain(self):
        """
        Returns agents brain as a dictionary.
        """
        return {
            "q_values": self.q_values,
            "learning_rate": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "training_error": self.training_error
        }
        
    def load_brain(self, brain_dict):
        self.q_values = brain_dict["q_values"]
        self.lr = brain_dict["learning_rate"]
        self.discount_factor = brain_dict["discount_factor"]
        self.epsilon = brain_dict["epsilon"]
        self.training_error = brain_dict["training_error"]

