import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from numpy import save
from numpy import load
import pathlib
from Libs.Controller.ControllerBase import *
from Libs.Environments.DataCollection import *
from Libs.Controller.MPNN import *


class ScalarDRLController(ControllerBase):
    def __init__(self, hidden_layers, env: DataCollection, learning_rate=0.001, gamma=0.95, epsilon_max=1.0,
                epsilon_min=0.01, epsilon_decay=0.0001, use_soft_max=True, soft_max_scaling=0.1):
        super().__init__()
        self.env = env  # reference to environment

        self.memory = deque(maxlen=10000)
        self.memory_top_reward = deque(maxlen=1000)
        self.best_sum_reward = -1e5
        self.sum_reward = 0
        self.memory_each_episode = []
        self.memory_virtual = deque(maxlen=10000)
        self.gamma = gamma  # discount rate
        self.epsilon_max = epsilon_max  # exploration rate
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.model = self._build_model(hidden_layers)
        self.target_model = self._build_model(hidden_layers)
        self.update_target_model()
        self.ep_counter = 0

        self.use_soft_max = use_soft_max
        self.soft_max_scaling = soft_max_scaling

    def _build_model(self, hidden_layers):
        layers = [tf.keras.Input(shape=(self.env.get_state_size(),))]
        for ls, ac in hidden_layers:
            if ac is None:
                layers.append(Dense(ls, kernel_initializer='he_normal')(layers[-1]))
            else:
                layers.append(Dense(ls, activation=ac, kernel_initializer='he_normal')(layers[-1]))
        layers.append(Dense(self.env.get_action_size(), activation='linear')(layers[-1]))
        model = tf.keras.Model(inputs=layers[0], outputs=layers[-1])
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, new_obs):
        self.memory.append(new_obs)
        _, _, reward, _, done = new_obs
        self.sum_reward += reward
        self.memory_each_episode.append(new_obs)
        if done:
            if self.sum_reward >= self.best_sum_reward:
                self.best_sum_reward = self.sum_reward
                for obs in self.memory_each_episode:
                    self.memory_top_reward.append(obs)
            self.memory_each_episode = []
            self.sum_reward = 0

    def memorize_virtual(self, new_obs):
        self.memory_virtual.append(new_obs)

    def get_soft_max_exploration(self, state):
        act_values = self.model(np.array([state]))
        act_values_scaled = tf.divide(act_values, tf.constant(self.soft_max_scaling, dtype=float))
        act_pr = tf.math.softmax(act_values_scaled).numpy()[0]
        return np.random.choice(range(self.env.get_action_size()), size=1, p=act_pr)[0]

    def get_epsilon_greedy_action(self, state):
        if (np.random.rand() <= self.epsilon) and (self.epsilon > 0):
            return random.randrange(self.env.get_action_size())
        act_values = self.model(np.array([state]))
        return np.argmax(act_values[0])  # returns action

    def act(self, state):
        if self.use_soft_max:
            return self.get_soft_max_exploration(state)
        else:
            return self.get_epsilon_greedy_action(state)

    def replay(self, batch_size, use_virtual_memory=0):
        # mini_batch: [state, action, reward, next_state, done]
        if use_virtual_memory == 0:
            # mini_batch = random.sample(self.memory, batch_size)
            # mini_batch.append(self.memory[-1])

            mini_batch = []
            if len(self.memory_top_reward) > 0:
                memory_0_rnd_idx = np.random.randint(0, len(self.memory_top_reward), batch_size)
            memory_1_rnd_idx = np.random.randint(0, len(self.memory), batch_size)
            memory_selection_pr = 0.2
            for i in range(batch_size):
                if (np.random.rand() < memory_selection_pr) and (len(self.memory_top_reward) > 0):
                    mini_batch.append(self.memory_top_reward[memory_0_rnd_idx[i]])
                else:
                    mini_batch.append(self.memory[memory_1_rnd_idx[i]])
        else:
            mini_batch = []
            memory_0_rnd_idx = np.random.randint(0, len(self.memory), batch_size)
            memory_1_rnd_idx = np.random.randint(0, len(self.memory_virtual), batch_size)
            memory_selection_pr = max(len(self.memory) / (len(self.memory) + len(self.memory_virtual)), 0.1)
            for i in range(batch_size):
                if np.random.rand() < memory_selection_pr:
                    mini_batch.append(self.memory[memory_0_rnd_idx[i]])
                else:
                    mini_batch.append(self.memory_virtual[memory_1_rnd_idx[i]])

        states = np.array([each[0] for each in mini_batch])
        actions = np.array([each[1] for each in mini_batch])
        rewards = np.array([each[2] for each in mini_batch])
        nextStates = np.array([each[3] for each in mini_batch])
        dones = np.array([each[4] for each in mini_batch])
        nextValues = np.max(self.target_model.predict(nextStates), axis=1)
        actualValue = np.where(dones, rewards, rewards + self.gamma * nextValues)

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            selectedActionValue = tf.math.reduce_sum(
                tf.multiply(self.model(states), tf.one_hot(actions, self.env.get_action_size())), axis=1)
            loss = tf.math.reduce_mean(tf.square(actualValue - selectedActionValue))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * self.ep_counter)

        self.ep_counter += 1

    def load(self, name):
        w_name = name + '_W'
        file = pathlib.Path(w_name + str('.index'))
        if file.exists():
            print("File exists")
            status = self.model.load_weights(w_name)
            mem_name = name + '_mem.npy'
            Memory_data = load(mem_name, allow_pickle=True)
            for data in Memory_data:
                self.memorize(data)
            self.update_target_model()
        else:
            print("File does not exist")

    def save(self, name):
        w_name = name + '_W'
        self.model.save_weights(w_name)

        mem_name = name + '_mem.npy'
        Memory_data = []
        for i in self.memory:
            Memory_data.append(i)
        save(mem_name, Memory_data)
