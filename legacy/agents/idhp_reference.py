import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import gym


class Actor(keras.Model):
    """
    Actor/Policy network
    """

    def __init__(self, config, env: gym.Env):
        super().__init__()
        self.env = env

        # Dimensions
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.shape[0]

        # Network
        self.actor = None
        self.layers_size = config["actor"]["layers"]
        if config["identity_init"]:
            self.kernel_initializer = keras.initializers.Identity()
        else:
            self.kernel_initializer = keras.initializers.truncated_normal(config["std_init"])

        # Create the actor
        self.setup()
        self.build(input_shape=(None, self.input_dim))

    def setup(self):
        """
        Build the policy model
        """

        # Input
        input = keras.Input(shape=(self.input_dim,))
        output = Flatten()(input)

        # Hidden layers
        for layer_size in self.layers_size:
            output = Dense(
                layer_size,
                activation=tf.nn.tanh,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
            )(output)

        # Output layer
        output = Dense(
            self.output_dim,
            activation=tf.nn.tanh,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(output)

        # Policy model
        self.actor = keras.Model(inputs=input, outputs=output)

    def call(self, obs):
        """
        Give the policy action for a given observation
        """

        action = self.actor(obs)

        return action

    @tf.function(experimental_relax_shapes=True)
    def get_policy_state_grad(self, state):
        """
        Compute the gradient of the policy w.r.t the state
        """

        # Get action inside tape context
        with tf.GradientTape(persistent=True) as state_tape:
            state_tape.watch(state)
            state_ = state / self.env.state_scale  # unscale the state
            obs_ = self.env.get_obs(state_)  # get observation w.r.t state inside the tape context
            action = self(obs_[np.newaxis])
            action_nodes = tf.split(action, self.output_dim, axis=1)

        # Compute state gradients for every action
        grads = []
        for i in range(len(action_nodes)):
            grads.append(state_tape.gradient(action_nodes[i], state))
        del state_tape
        dpidx = tf.stack(grads, axis=0)

        return dpidx
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten
import gym


class Critic(keras.Model):
    """
    Critic state-value network
    """

    def __init__(self, config, env: gym.Env):
        super().__init__()

        # Dimensions
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.state.shape[0]

        # Network
        self.critic = None
        self.layers_size = config["critic"]["layers"]
        self.kernel_initializer = keras.initializers.truncated_normal(config["std_init"])

        # Create the critic network
        self.setup()
        self.build(input_shape=(None, self.input_dim))

    def setup(self):
        """
        Build the critic model
        """

        # Input
        input = keras.Input(shape=(self.input_dim,))
        output = Flatten()(input)

        # Hidden layers
        for layer_size in self.layers_size:
            output = Dense(
                layer_size,
                activation=tf.nn.tanh,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
            )(output)

        # Output layer
        output = Dense(
            self.output_dim,
            activation=None,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(output)

        # Critic model
        self.critic = keras.Model(inputs=input, outputs=output)

    def call(self, obs):
        """
        Give the state-value for given observation
        """

        return self.critic(obs)

    @tf.function(experimental_relax_shapes=True)
    def soft_update(self, source: keras.Model, tau):
        """
        Update parameters from a source network
        """

        source = source.trainable_weights
        target = self.trainable_weights

        for target_var, source_var in zip(target, source):
            target_var.assign((1.0 - tau) * target_var + tau * source_var)
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import numpy as np
from tqdm import tqdm
import gym

from agents.idhp import Actor
from agents.idhp import Critic
from agents.idhp import Model
from tools import scale_action, set_random_seed


class IDHP:
    """
    Incremental Dual Heuristic Programming (IDHP)
    On-Policy Online Reinforcement Learning using Incremental Model
    """

    def __init__(self, config, env: gym.Env):
        set_random_seed(config["seed"])
        self.config = config

        # Environment
        self.env = env

        # Actor
        self.actor = Actor(config, self.env)

        # Critic
        self.critic = Critic(config, self.env)
        self.critic_target = Critic(config, self.env)

        # Incremental model
        self.model = Model(config, self.env)

        # Training
        self.lr_actor_high = config["actor"]["lr_high"]
        self.lr_actor_low = config["actor"]["lr_low"]
        self.lr_critic_high = config["critic"]["lr_high"]
        self.lr_critic_low = config["critic"]["lr_low"]
        self.lr_actor = tf.Variable(self.lr_actor_high, trainable=False)
        self.lr_critic = tf.Variable(self.lr_critic_high, trainable=False)
        self.lr_adapt = config["lr_adapt"]
        self.lr_adapt_warmup = config["lr_adapt_warmup"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.reward_scale = config["reward_scale"]
        self.t = 0

        # Optimizers
        self.actor_optimizer = SGD(self.lr_actor)
        self.critic_optimizer = SGD(self.lr_critic)

        # Logging
        self.actor_weights_history = []
        self.critic_weights_history = []
        self.F_history = []
        self.G_history = []
        self.cov_history = []
        self.epsilon_history = []

    @tf.function(experimental_relax_shapes=True)
    def call(self, obs):
        """
        Give the policy's action for a given observation
        """

        action = self.actor(obs[np.newaxis])
        action = tf.reshape(action, [-1])  # remove batch dimension

        return action

    def learn(self):
        """
        Training loop (online, single episode)
        """

        # Initialize target networks
        self.critic_target.soft_update(source=self.critic, tau=1.0)

        # Initialize environment
        obs, info = self.env.reset()
        state = info["state"]
        state_prev = None
        action_prev = None

        # Start (online) training loop, single episode
        for step in (bar := tqdm(range(self.env.task.num_timesteps))):
            bar.set_description("Training IDHP")

            # Take action
            action = self.call(obs)
            obs_next, _, _, info = self.env.step(scale_action(action, self.env.action_space))
            state_next = info["state"]
            reward_grad = info["reward_grad"] * self.reward_scale
            lr_thresh = info["lr_thresh"]
            self.t += 1

            # Adaptive learning rate
            if self.lr_adapt and self.t >= self.lr_adapt_warmup:
                if lr_thresh:
                    if self.lr_actor != self.lr_actor_high:
                        self.lr_actor.assign(self.lr_actor_high)
                    if self.lr_critic != self.lr_critic_high:
                        self.lr_critic.assign(self.lr_critic_high)
                else:
                    if self.lr_actor != self.lr_actor_low:
                        self.lr_actor.assign(self.lr_actor_low)
                    if self.lr_critic != self.lr_critic_low:
                        self.lr_critic.assign(self.lr_critic_low)

            # Update networks
            self.learn_step(obs, obs_next, state, reward_grad, self.model.F, self.model.G)

            # Update model
            if step > 1:
                self.model.update(state - state_prev, action - action_prev, state_next - state)

            # Update samples
            obs = obs_next
            action_prev = action
            state_prev = state
            state = state_next

            # Logging
            self.actor_weights_history.append(self.actor.get_weights())
            self.critic_weights_history.append(self.critic.get_weights())
            self.F_history.append(self.model.F)
            self.G_history.append(self.model.G)
            self.cov_history.append(self.model.Cov)
            self.epsilon_history.append(self.model.epsilon)

    @tf.function(experimental_relax_shapes=True)
    def learn_step(self, obs, obs_next, state, reward_grad, F, G):
        """
        Updates the actor and critic
        """

        with tf.GradientTape(persistent=True) as tape:

            # Actor call
            action = self.actor(obs[np.newaxis])

            # Critic call
            lmbda = self.critic(obs[np.newaxis])
            lmbda_next = self.critic_target(obs_next[np.newaxis])

        # Actor loss
        actor_loss_grad = -(reward_grad + self.gamma * lmbda_next) @ G
        actor_loss_grad = tape.gradient(
            action, self.actor.trainable_weights, output_gradients=actor_loss_grad
        )

        # Actor update
        self.actor_optimizer.apply_gradients(zip(actor_loss_grad, self.actor.trainable_weights))

        # Critic loss
        dpidx = self.actor.get_policy_state_grad(state)
        td_err_ds = (reward_grad + self.gamma * lmbda_next) @ (F + G @ dpidx) - lmbda
        critic_loss_grad = -td_err_ds
        critic_loss_grad = tape.gradient(
            lmbda, self.critic.trainable_weights, output_gradients=critic_loss_grad
        )

        # Critic update
        self.critic_optimizer.apply_gradients(zip(critic_loss_grad, self.critic.trainable_weights))
        self.critic_target.soft_update(source=self.critic, tau=self.tau)

        del tape
import numpy as np
import gym


class Model:
    """
    Recursive Least Squares (RLS) incremental environment model
    """

    def __init__(self, config, env: gym.Env):
        self.env = env

        # Dimensions
        self.state_size = env.state.shape[0]
        self.action_size = env.action_space.shape[0]

        # Config
        self.gamma = config["model"]["gamma"]

        # Initialize measurement matrix
        self.X = np.ones((self.state_size + self.action_size, 1))

        # Initialise parameter matrix
        self.Theta = np.zeros((self.state_size + self.action_size, self.state_size))

        # Initialize covariance matrix
        self.Cov0 = config["model"]["cov0"] * np.identity(self.state_size + self.action_size)
        self.Cov = self.Cov0

        # Initial innovation (prediction error)
        self.epsilon = np.zeros((1, self.state_size))
        self.Cov_reset = False

    @property
    def F(self):
        return np.float32(self.Theta[: self.state_size, :].T)

    @property
    def G(self):
        return np.float32(self.Theta[self.state_size :, :].T)

    def update(self, state, action, state_next):
        """
        Update RLS parameters based on state-action sample
        """

        # Predict next state
        state_next_pred = self.predict(state, action)

        # Error
        self.epsilon = (np.array(state_next)[np.newaxis].T - state_next_pred).T

        # Intermediate computations
        CovX = self.Cov @ self.X
        XCov = self.X.T @ self.Cov
        gammaXCovX = self.gamma + XCov @ self.X

        # Update parameter matrix
        self.Theta = self.Theta + (CovX @ self.epsilon) / gammaXCovX

        # Update covariance matrix
        self.Cov = (self.Cov - (CovX @ XCov) / gammaXCovX) / self.gamma

        # Check if Cov needs reset
        epsilon_tresh = np.array(self.env.model_eps_tresh)
        if self.Cov_reset == False:
            if np.sum(np.greater(np.abs(self.epsilon) / self.env.state_scale, epsilon_tresh)) == 1:
                self.Cov_reset = True
                self.Cov = self.Cov0
        elif self.Cov_reset == True:
            if np.sum(np.greater(np.abs(self.epsilon) / self.env.state_scale, epsilon_tresh)) == 0:
                self.Cov_reset = False

    def predict(self, state, action):
        """
        Predict next state based on RLS model
        """

        # Set measurement matrix
        self.X[: self.state_size] = np.array(state)[np.newaxis].T
        self.X[self.state_size :] = np.array(action)[np.newaxis].T

        # Predict next state
        state_next_pred = (self.X.T @ self.Theta).T

        return state_next_pred
