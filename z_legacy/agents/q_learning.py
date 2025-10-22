from agents.base_agent import Agent
import numpy as np
from collections import defaultdict
from functools import partial
from utils.checkpoint_utils import safe_eval_key  # Import the safe evaluation function

class QLearning(Agent):
    def update(
        self,
        obs: tuple,
        action: tuple,
        reward: float,
        terminated: bool,
        next_obs: tuple,
    ):
        obs = self.obs_discretizer.discretize(obs)
        next_obs = self.obs_discretizer.discretize(next_obs)
        action = self.action_discretizer.discretize(action)
        assert all(
            0 <= a < n for a, n in zip(action, self.action_discretizer.space.nvec)
        ), (
            f"Discretized action {action} out of bounds for nvec {self.action_discretizer.space.nvec}"
        )
        flat_action = np.ravel_multi_index(action, self.action_discretizer.space.nvec)

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
        Returns agent's brain as a dictionary for serialization.
        """
        # Convert keys to strings for serialization
        serializable_q = {str(k): v for k, v in self.q_values.items()}
        return {
            "q_values": serializable_q,
            "learning_rate": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "training_error": self.training_error,
        }

    def load_brain(self, brain_dict):
        """
        Load agent's brain from a dictionary.
        """
        loaded_q = brain_dict["q_values"]

        # ✅ Safely convert keys from stringified tuples back to real tuples
        restored_q = {
            safe_eval_key(k): np.array(v) if not isinstance(v, np.ndarray) else v
            for k, v in loaded_q.items()
        }

        # ✅ Restore with picklable default factory
        self.q_values = defaultdict(
            partial(np.zeros, self.num_actions), restored_q
        )

        self.lr = brain_dict["learning_rate"]
        self.discount_factor = brain_dict["discount_factor"]
        self.epsilon = brain_dict["epsilon"]
        self.training_error = brain_dict["training_error"]