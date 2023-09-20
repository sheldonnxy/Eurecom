from abc import ABC, abstractmethod
from .ChannelDefines import *
from ..CitySimulator.CityConfigurator import *


class RadioChannelBase(ABC):
    def __init__(self):
        self.ch_param = ChannelParamStr()

    def update_ch_param(self, ch_param):
        self.ch_param = ch_param

    def get_ch_param(self):
        return self.ch_param

    def get_measurement_from_users(self, city, q_poses, user_poses, sampling_resolution=None):
        sampled_q_poses = q_poses.copy()
        num_q = q_poses.shape[0]

        if (sampling_resolution is not None) and (num_q > 1):
            sampled_q_poses = generate_uav_trajectory_from_points(q_poses=q_poses,
                                                                  sampling_resolution=sampling_resolution)
        return self.collect_measurements(city, sampled_q_poses, user_poses)

    def get_measurement_from_user(self, city, q_pose, user_pose, sampling_resolution=None):
        sampled_q_pose = q_pose.copy()
        num_q = q_pose.shape[0]

        return self.collect_measurement(city, sampled_q_pose, user_pose)

    @abstractmethod
    def collect_measurements(self, city, q_poses, user_poses=None):
        return [[RadioMeasurementStr()]]

    @abstractmethod
    def collect_measurement(self, city, q_pose, user_pose=None):
        return [[RadioMeasurementStr()]]

    def get_radio_map(self, city, q_height=None, ue_poses=None, resolution=10):
        x_ = np.arange(0, city.urban_config.map_x_len, resolution)
        y_ = np.arange(0, city.urban_config.map_y_len, resolution)

        radio_map = RadioMapStr(shape=[len(ue_poses), len(x_), len(y_)])

        for i_x, x in enumerate(x_):
            for i_y, y in enumerate(y_):
                uav_pose = np.reshape(np.r_[np.array(x), np.array(y), q_height], newshape=(1, 3))
                measurements = self.get_measurement_from_users(city, uav_pose, ue_poses)
                for k, val in enumerate(measurements[0]):
                    radio_map.los[k, i_x, i_y] = val.los_status
                    radio_map.rssi_db[k, i_x, i_y] = val.rssi_db
                    radio_map.ch_gain_db[k, i_x, i_y] = val.ch_gain_db
                    radio_map.capacity[k, i_x, i_y] = val.ch_capacity
        return radio_map
