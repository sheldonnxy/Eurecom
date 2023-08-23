from abc import ABC, abstractmethod

import numpy as np
from ..ChannelModel.ChannelDefines import *
from ..CitySimulator.CityConfigurator import *


class EnvironmentBase(ABC):
    def __init__(self):
        self.env_param = None
        self.agent_pose = np.zeros(shape=[1, 3])
        self.agent_state = None
        self.agent_info = None

    @abstractmethod
    def reset(self):
        pass

    # def step(self, action):
    #     self.agent_mobility_model(action)
    #     collected_meas = self.env_param.radio_ch_model.get_measurement_from_users(self.env_param.city_model,
    #                                                                               self.agent_pose,
    #                                                                               self.env_param.device_position)
    #     collected_data, idx = self.comm_step(collected_meas)
    #     agent_state = self.agent_state_update(action, measurements=collected_meas)
    #     reward, done = self.agent_reward_done(action, measurements=collected_data)
    #
    #     return reward, agent_state, done, self.agent_info, idx

    def step(self, actions):
        raise NotImplementedError

    @abstractmethod
    def comm_step(self, collected_meas, d_id, model):
        pass

    # @abstractmethod
    # def agent_mobility_model(self, action):
    #     pass

    # @abstractmethod
    # def get_state_vec(self):
    #     pass

    # @abstractmethod
    # def agent_state_update(self, action=None, measurements=None):
    #     pass

    # @abstractmethod
    # def agent_reward_done(self, action=None, measurements=None):
    #     pass

    # TODO: add these functions to DtaCollection Env

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        """ Return the global state """
        raise NotImplementedError

    def get_state_size(self):
        """ Return the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

