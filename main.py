from tqdm import tqdm
import gymnasium as gym
from matplotlib import pyplot as plt   
from env import InvertedPendulumEnv
from agent import QLearning
import numpy as np
from utils.discretizer import UniformTileCoding
from utils.plots import plot_best_action, plot_test_episode

# hyperparameters
learning_rate = 0.05
n_epsiodes = 1000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_epsiodes / 2)
final_epsilon = 0.1

env = InvertedPendulumEnv()
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_epsiodes)

agent = QLearning(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=0.95,
    state_discretizer=UniformTileCoding(
        env.observation_space,
        bins=(10, 10)
    ),
    action_discretizer=UniformTileCoding(
        env.action_space,
        bins=(10,)
    )
)

for episode in tqdm(range(n_epsiodes)):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)
        obs = next_obs
        done = terminated or truncated

    agent.decay_epsilon()

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500 episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)

axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()

plot_best_action(agent, agent.agent_state_space)

plot_test_episode(agent, env, n_steps=1000)