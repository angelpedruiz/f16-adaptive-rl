import gymnasium as gym
import numpy as np
from tqdm import tqdm
from agents.base_agent import QLearning
from utils.discretizer import UniformTileCoding
from utils.plots import reward_length_learning_error_plot, plot_episode_lunar_lander

### HYPERPARAMETERS ###
learning_rate = 0.4
n_epsiodes = 600
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_epsiodes / 2)
final_epsilon = 0.1
#######################

### AGENT INITIALIZATION ###
env_name = "LunarLander-v3"
env = gym.make(env_name, continuous=True, gravity = -9.81, enable_wind=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_epsiodes)

agent = QLearning(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=0.95,
    obs_discretizer=UniformTileCoding(
        env.observation_space, bins=(10, 10, 10, 10, 10, 10, 10, 10)
    ),
    action_discretizer=UniformTileCoding(env.action_space, bins=(10, 10))
)
#############################

### TEST ENV ###
# episodes = 3
# for episode in range(episodes):
#     obs, info = env.reset()
#     done = False
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         next_obs, reward, terminated, truncated, info = env.step(action)
#         obs = next_obs
#         done = terminated or truncated
#     print('Episode:{} Score:{}'.format(episode, info['episode']['r']))
# env.close()
################

### TRAINING LOOP ###
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
#####################

### RESULTS PLOTTING ###
reward_length_learning_error_plot(env, agent, rolling_length=50)
plot_episode_lunar_lander(agent, env=gym.make(env_name, continuous=True, gravity = -9.81, enable_wind=False, render_mode='human'))
env.close()
#######################