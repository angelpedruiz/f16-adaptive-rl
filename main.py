from tqdm import tqdm
import gymnasium as gym
from matplotlib import pyplot as plt
from env import LinearModelF16
from agent import QLearning
from data.LinearF16SS import A_long_hi as A, B_long_hi as B
from utils.discretizer import UniformTileCoding
from utils.plots import reward_length_learning_error_plot, plot_test_episode

# hyperparameters 
learning_rate = 0.05
n_epsiodes = 10
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_epsiodes / 2)
final_epsilon = 0.1

env = LinearModelF16(A, B)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_epsiodes)

agent = QLearning(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=0.95,
    obs_discretizer=UniformTileCoding(
        env.observation_space , bins=(10, 10, 10, 10, 10, 10, 10, 10)
    ),
    action_discretizer=UniformTileCoding(env.action_space, bins=(10, 10)),
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

# plot_best_action(agent, agent.agent_state_space)

plot_test_episode(agent, env, n_steps=1000)
#reward_length_learning_error_plot(env, agent, rolling_length=50)
