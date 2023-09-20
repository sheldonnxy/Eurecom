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

    def step(self, action):
        self.agent_mobility_model(action)
        collected_meas = self.env_param.radio_ch_model.get_measurement_from_users(self.env_param.city_model,
                                                                                  self.agent_pose,
                                                                                  self.env_param.user_pose)
        agent_state = self.agent_state_update(action, measurements=collected_meas)
        reward, done = self.agent_reward_done(action, measurements=collected_meas)

        return reward, agent_state, done, self.agent_info

    @abstractmethod
    def agent_mobility_model(self, action):
        pass

    @abstractmethod
    def agent_state_update(self, action=None, measurements=None):
        pass

    @abstractmethod
    def agent_reward_done(self, action=None, measurements=None):
        pass
