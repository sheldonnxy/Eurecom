from .EnvironmentBase import *


class EnvParamStr:
    def __init__(self):
        self.city_model = CityConfigurator()
        self.radio_ch_model = ChannelParamStr()
        self.user_pose = np.zeros(shape=[1, 3])
        self.battery_budget = None
        self.step_size = None
        self.start_pose = None
        self.end_pos = None
        self.safety_controller_flag = False


class AgentStateModel:
    def __init__(self, agent_pose_index, user_pose_index, battery_budget):
        self.agent_pose_index = agent_pose_index
        self.user_pose_index = user_pose_index
        self.battery = battery_budget
        self.num_users = user_pose_index.shape[0]

    def get_state_vec(self):
        agent_pose_flat = self.agent_pose_index.flatten()

        relative_uav_user_pose = np.repeat(self.agent_pose_index, self.num_users, axis=0) - self.user_pose_index
        relative_uav_user_pose_flat = relative_uav_user_pose.flatten()
        battery = np.array([self.battery])

        agent_state = np.r_[agent_pose_flat, relative_uav_user_pose_flat, battery]
        return agent_state


class AgentMobilityModel:
    def __init__(self):
        # actions: [hover, north, west, south, east], Note that the height is fixed
        self.action_space = np.array([[0, 0, 0],
                                      [0, 1, 0],
                                      [-1, 0, 0],
                                      [0, -1, 0],
                                      [1, 0, 0],
                                      ])

        self.power_consumption_table = np.ones(shape=[len(self.action_space)])
        self.power_consumption_table[0] *= 0.5

    def agent_move(self, current_pose, current_battery_status, action):
        next_pose = current_pose + self.action_space[int(action)]
        next_battery_status = current_battery_status - self.power_consumption_table[int(action)]
        return next_pose, next_battery_status


class EnvInfoStr:
    def __init__(self):
        self.agent_pose = 0
        self.collected_data = 0
        self.battery_budget = 0


class DataCollection(EnvironmentBase):
    def __init__(self, env_param):
        super().__init__()
        self.env_param = env_param

        # Additional variables specific to this environment ...............
        self.agent_pose_index = np.zeros_like(self.agent_pose)
        self.agent_battery_budget = 0
        self.total_collected_data = 0
        self.agent_action_model = AgentMobilityModel()
        self.reward_penalty = 0

        self.reset()

    def reset(self):
        self.agent_pose_index = self.pose_to_index(self.env_param.start_pose)
        self.agent_pose = self.index_to_pose(self.agent_pose_index)
        self.total_collected_data = 0
        self.agent_battery_budget = self.env_param.battery_budget
        user_pose_index = self.pose_to_index(self.env_param.user_pose)
        self.agent_state = AgentStateModel(self.agent_pose_index, user_pose_index, self.agent_battery_budget)
        self.agent_info = EnvInfoStr()
        self.reward_penalty = 0

        return self.agent_state.get_state_vec()

    def get_action_size(self):
        return int(len(self.agent_action_model.action_space))

    def get_state_size(self):
        return int(len(self.agent_state.get_state_vec()))

    def agent_mobility_model(self, action):
        if self.env_param.safety_controller_flag:
            action, self.reward_penalty = self.safety_controller(action)
        agent_pose_index, self.agent_battery_budget = self.agent_action_model.agent_move(self.agent_pose_index
                                                                                         , self.agent_battery_budget
                                                                                         , action)
        self.agent_pose_index, _ = self.check_uav_pos(agent_pose_index)
        self.agent_pose = self.index_to_pose(self.agent_pose_index)

    def agent_state_update(self, action=None, measurements=None):
        self.agent_state.agent_pose_index = self.agent_pose_index
        self.agent_state.battery = self.agent_battery_budget
        return self.agent_state.get_state_vec()

    def agent_reward_done(self, action=None, measurements=None):
        done = 0
        reward = 0
        new_data = 0
        for data_arr in measurements[0]:
            new_data += data_arr.ch_capacity
        new_data = new_data / len(measurements[0])
        self.total_collected_data += new_data

        end_pose_index = self.pose_to_index(self.env_param.end_pos)
        if np.array_equal(self.agent_pose_index, end_pose_index) and self.agent_battery_budget >= 0:
            done = 1
            reward = new_data
        elif self.agent_battery_budget <= 0:
            done = 1
            distance_to_end_pose = np.linalg.norm(end_pose_index - self.agent_pose_index)
            reward = -50 + 10 / (distance_to_end_pose + 1)

        reward = reward + self.reward_penalty

        self.agent_info.collected_data = self.total_collected_data
        self.agent_info.agent_pose = self.agent_pose
        self.agent_info.battery_budget = self.agent_battery_budget

        return reward, done

    def pose_to_index(self, pos):
        index = pos / self.env_param.step_size
        index[:, -1] = pos[:, -1]
        return index.astype(int)

    def index_to_pose(self, index):
        pose = (index * self.env_param.step_size)
        pose[:, -1] = index[:, -1]
        return pose

    def check_uav_pos(self, pos_idx):
        index = pos_idx.copy()
        max_x_idx = int(self.env_param.city_model.urban_config.map_x_len/self.env_param.step_size)
        max_y_idx = int(self.env_param.city_model.urban_config.map_y_len/self.env_param.step_size)
        index[0, 0] = max(0, pos_idx[0, 0])
        index[0, 1] = max(0, pos_idx[0, 1])
        index[0, 0] = min(max_x_idx, index[0, 0])
        index[0, 1] = min(max_y_idx, index[0, 1])

        invalid_uav_pose = (1 - np.array_equal(pos_idx, index))
        return index, invalid_uav_pose

    def safety_controller(self, action):
        end_pose_index = self.pose_to_index(self.env_param.end_pos)
        dist_to_dest = end_pose_index - self.agent_pose_index
        horizontal_steps = dist_to_dest[0, 0]
        vertical_steps = dist_to_dest[0, 1]
        horizontal_action_type = (horizontal_steps > 0) * 4 + (horizontal_steps < 0) * 2
        vertical_action_type = (vertical_steps > 0) * 1 + (vertical_steps < 0) * 3
        horizontal_actions = np.ones(shape=[int(abs(horizontal_steps))]) * horizontal_action_type
        vertical_actions = np.ones(shape=[int(abs(vertical_steps))]) * vertical_action_type

        required_actions_to_end = np.r_[horizontal_actions, vertical_actions]
        required_battery_to_end = 0
        for ac in required_actions_to_end:
            required_battery_to_end += self.agent_action_model.power_consumption_table[int(ac)]

        extra_battery_to_end = self.agent_battery_budget - required_battery_to_end
        required_actions_to_end = np.random.choice(required_actions_to_end, len(required_actions_to_end))
        reward_penalty = 0

        if action not in required_actions_to_end:
            if extra_battery_to_end == 0:
                action = required_actions_to_end[0]
                reward_penalty = -0.08
            elif extra_battery_to_end < 2:
                action = 0
                reward_penalty = -0.08

        return int(action), reward_penalty
