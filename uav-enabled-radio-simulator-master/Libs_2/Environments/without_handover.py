import numpy as np
import os

from .EnvironmentBase import *
from .IoTDevice import DeviceList


class AgentModel:
    def __init__(self,
                 start_pose=np.array([[0, 0, 0]]),
                 end_pose=np.array([[0, 0, 0]]),
                 battery_budget=50.0
                 ):
        # hover, north, west, south, east, no-op when arriving destination
        self.action_space = np.array([[0, 0, 0],
                                      [0, 1, 0],
                                      [-1, 0, 0],
                                      [0, -1, 0],
                                      [1, 0, 0],
                                      [0, 0, 0],
                                      ])
        self.power_consumption_table = np.ones(shape=[len(self.action_space)])
        self.power_consumption_table[0] *= 0.5
        self.power_consumption_table[len(self.action_space) - 1] = 0.0
        self.battery_budget = battery_budget
        self.battery = 0
        # self.collected_data = None
        self.step_size = 20

        self.start_pose = start_pose
        self.end_pose = end_pose
        self.current_pose = None
        self.current_pose_index = None
        self.done = False
        self.collected_device = np.array([-1])

    # def get_obs_agent(self):
    #     pass
    def agent_move(self, action):
        self.current_pose_index += self.action_space[int(action)]
        self.current_pose = self.index_to_pose(self.current_pose_index)
        self.battery -= self.power_consumption_table[int(action)]

    def pose_to_index(self, pos):
        index = pos / self.step_size
        index[:, -1] = pos[:, -1]
        return index.astype(int)

    def index_to_pose(self, index):
        pose = (index * self.step_size)
        pose[:, -1] = index[:, -1]
        return pose


class EnvInfoStr:
    def __init__(self):
        self.agent_pose = 0
        self.collected_data = 0
        self.battery = 0


class DataCollection(EnvironmentBase):
    def __init__(self, args):
        super().__init__()
        # environment information
        self.args = args
        self.city = args.city
        self.step_size = args.step_size
        # the second dimension is x and the first dimension is y
        self.max_index_x = int(self.city.urban_config.map_x_len / self.step_size)
        self.max_index_y = int(self.city.urban_config.map_y_len / self.step_size)
        self.radio_ch_model = args.radio_ch_model
        self.link_status = self.init_link_status()
        # self.battery_budget = args.battery_budget

        self.safety_controller_flag = args.safety_controller_flag

        # Device information
        self.device_params = args.device_params
        self.device_position = args.device_position
        self.n_devices = self.device_position.shape[0]
        self.device_list = DeviceList(params=self.device_params)
        self.device_data = np.array([device.remaining_data for device in self.device_list.devices]).reshape(
            (len(self.device_position), 1)).copy()

        # agent information
        self.agent_params = args.agent_params
        self.start_pose = args.agent_params['start_pose']
        self.end_pose = args.agent_params['end_pose']
        self.n_agents = args.agent_params['agent_num']
        # initialize agents
        self.agents = [AgentModel(args.agent_params['start_pose'][i], args.agent_params['end_pose'][i],
                                  args.agent_params['battery_budget'][i]) for i in
                       range(args.agent_params['agent_num'])]

        # action
        self.n_actions = 6
        # self.n_actions = self.n_actions_move + self.n_devices
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # global state names of attributes
        self.ally_state_sttr_names = [
            "battery",
            "extra_battery_to_end",
            "pos_index_x",
            "pos_index_y",
            "done",
        ]
        self.device_state_attr_name = [
            "remaining_data",
            "pos_index_x",
            "pos_index_y",
        ]

        # observation dimension
        self.obs_move_feats_size = 6
        self.obs_device_feats_size = (self.n_devices, 6)
        self.obs_ally_feats_size = (self.n_agents - 1, 6)
        self.obs_own_feats_size = 4

        # sight range between UAV and device and UAVs
        self.device_sight_range = 10
        self.uav_sight_range = 10

        # reward setting
        self.reward_scale = 1000
        self.movement_penalty = 0.2
        self.reward = 0
        self.total_collected_data = 0
        self.reward_penalty = 0

        self._episode_steps = 0
        self.episode_limit = 100

        # whether add last_action or extra_battery_to_end to global state
        self.state_last_action = True
        self.state_extra_battery_to_end = False

        self.reset()

        self.data_scale = 20000
        self.dis_scale = 200
        self.battery_scale = 50

    def reset(self):
        self._episode_steps = 0
        # reset UAVs
        for i, agent in enumerate(self.agents):
            agent.current_pose = agent.start_pose.copy()
            agent.current_pose_index = agent.pose_to_index(agent.current_pose)
            agent.collected_data = 0
            agent.battery = agent.battery_budget.copy()
            agent.done = False
            # agent.collected_data = []

        # reset devices
        for device in self.device_list.devices:
            device.remaining_data = device.data.copy()

        self.agent_info = EnvInfoStr()
        self.total_collected_data = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        return

    def get_action_size(self):
        return int(len(self.agent_action_model.action_space))

    def comm_step(self, collected_meas, d_id):
        throughput = collected_meas.ch_capacity
        collect_data = self.device_list.collect_data(throughput, d_id)
        return collect_data

    def agent_reward_done(self, action=None, measurements=None):
        done = 0
        reward = 0
        new_data = measurements
        # for data_arr in measurements[0]:
        #     new_data += data_arr.ch_capacity
        # new_data = new_data / len(measurements[0])
        reward = new_data
        self.total_collected_data += new_data

        end_pose_index = self.pose_to_index(self.env_param.end_pos)
        if np.array_equal(self.agent_pose_index, end_pose_index) and self.battery >= 0:
            done = 1
        elif self.battery <= 0:  # TODO: useless decision because of the safety controller
            done = 1  # TODO: try to change the reward to get rid of the safety controller
            distance_to_end_pose = np.linalg.norm(end_pose_index - self.agent_pose_index)
            reward = -50 + 10 / (distance_to_end_pose + 1)

        # TODO: should add penalty to each action in order to force the agent go to destination after collecting all
        #  the data
        reward = reward + self.reward_penalty

        self.agent_info.collected_data = self.total_collected_data
        self.agent_info.agent_pose = self.agent_pose
        self.agent_info.battery = self.battery

        return reward, done

    def pose_to_index(self, pos):
        index = pos / self.step_size
        index[:, -1] = pos[:, -1]
        return index.astype(int)

    def index_to_pose(self, index):
        pose = (index * self.step_size)
        pose[:, -1] = index[:, -1]
        return pose

    def check_uav_pos(self, pos_idx):
        """Check whether the position of the UAV is inside the map after taking an action"""
        x, y = pos_idx[0, 0], pos_idx[0, 1]
        if 0 <= x <= self.max_index_x and 0 <= y <= self.max_index_y:
            return 1
        else:
            return 0

    def safety_controller(self, agent):

        avail_movement_actions = [0] * self.n_actions

        # # the first step, let UAV fly away from the start point
        # if self._episode_steps == 0:
        #     avail_movement_actions = [1] * self.n_actions
        #     avail_movement_actions[0] = 0
        #     avail_movement_actions[self.n_actions - 1] = 0
        #     extra_battery_to_end = agent.battery_budget.copy()
        #     return avail_movement_actions, extra_battery_to_end

        end_pose_index = self.pose_to_index(agent.end_pose)
        dist_to_end = end_pose_index - agent.current_pose_index
        horizontal_steps = dist_to_end[0, 0]
        vertical_steps = dist_to_end[0, 1]
        horizontal_action_type = (horizontal_steps > 0) * 4 + (horizontal_steps < 0) * 2
        vertical_action_type = (vertical_steps > 0) * 1 + (vertical_steps < 0) * 3

        horizontal_actions = np.ones(shape=[int(abs(horizontal_steps))]) * horizontal_action_type
        vertical_actions = np.ones(shape=[int(abs(vertical_steps))]) * vertical_action_type

        action_type = [horizontal_action_type, vertical_action_type]

        required_actions_to_end = np.concatenate([horizontal_actions, vertical_actions])
        required_battery_to_end = 0
        for ac in required_actions_to_end:
            required_battery_to_end += agent.power_consumption_table[int(ac)]

        extra_battery_to_end = agent.battery - required_battery_to_end

        if extra_battery_to_end == 0:
            for i, action_index in enumerate(action_type):
                if action_index != 0:
                    avail_movement_actions[action_index] = 1
        elif extra_battery_to_end < 2:
            avail_movement_actions[0] = 1
        else:
            avail_movement_actions = [1] * (self.n_actions - 1) + [0]
            # required_actions_to_end = np.random.choice(required_actions_to_end, len(required_actions_to_end))
        # reward_penalty = 0

        # if action not in required_actions_to_end:
        #     if extra_battery_to_end == 0:
        #         action = required_actions_to_end[0]
        #         reward_penalty = -0.08
        #     elif extra_battery_to_end < 2:
        #         action = 0
        #         reward_penalty = -0.08

        # return int(action), reward_penalty
        return avail_movement_actions, extra_battery_to_end

    def step(self, actions):
        """ Returns reward, terminated, info """
        self.last_action = np.eye(self.n_actions)[np.array(actions)]
        agent_collected_data = []
        movement_reward = []
        terminated = False
        info = []

        for a_id, action in enumerate(actions):
            agent = self.get_agent_by_id(a_id)
            # if the agent arrives the terminal position and choose the non-op action, then this agent is done
            if action == len(agent.action_space) - 1:
                agent.done = True
            if agent.battery > 0 and agent.done is False:
                # get the connected device's index
                a = self.get_avail_device(a_id)
                device_id = np.nonzero(self.get_avail_device(a_id))[0]
                # if there is a device connecting to this agent, then collect data from this device
                if len(device_id) != 0:
                    device = self.device_list.get_device(int(device_id))
                    collected_meas = self.radio_ch_model.get_measurement_from_user(self.city,
                                                                                   agent.current_pose,
                                                                                   device.position)
                    collected_data = self.comm_step(collected_meas, int(device_id))
                    agent.collected_device = device_id
                    agent_collected_data.append(collected_data)
                    # agent_collected_data.append(np.sum(agent.collected_data))
                else:
                    agent.collected_device = np.array([-1])
                agent.agent_move(action)
                movement_reward.append(self.movement_penalty * agent.power_consumption_table[int(action)])
            else:
                agent.collected_device = np.array([-1])

        total_data = np.sum(agent_collected_data)
        self.total_collected_data += total_data
        data_reward = total_data / self.reward_scale
        reward = data_reward - np.sum(movement_reward)

        # determine whether all the agents arrive the destination and are done
        n_done = 1
        for idx, agent in enumerate(self.agents):
            n_done *= agent.done
        if n_done == 1 or self._episode_steps >= self.episode_limit:
            terminated = True
        self._episode_steps += 1
        return reward, terminated, info
        # return reward, terminated, info

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for each agent:
        - agent movement features (where it can move to)
        - device features (available_to_access, SNR_db, remaining data, distance, relative_x, relative_y)
        _ ally features (communicate, SNR_db, remaining battery, distance, relative_x, relative_y)
        - agent feature (remaining battery, extra battery to destination, relative_x to destination, relative_y to destination)
        """
        agent = self.agents[agent_id]

        move_feats_dim = self.obs_move_feats_size
        device_feats_dim = self.obs_device_feats_size
        ally_feats_dim = self.obs_ally_feats_size
        own_feats_dim = self.obs_own_feats_size

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        device_feats = np.zeros(device_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        # if the UAV still has enough battery and doesn't arrive destination
        # TODO: whether show the location of the UAV with "done"
        if agent.battery > 0 and agent.done is False:

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            device_access = self.get_avail_device(agent_id)
            for m in range(self.n_actions):
                move_feats[m] = avail_actions[m]

            # Device features
            for d_id, device in enumerate(self.device_list.devices):
                pos = device.position
                snr_db = self.radio_ch_model.snr_measurement(self.city, pos, agent.current_pose, 'device')
                # the signal strength is enough to let the UAV get access to the info of the device
                if snr_db >= self.device_sight_range:
                    device_feats[d_id, 0] = device_access[d_id]  # whether collecting data from this device
                    # TODO: /self.device_sight_range or another value? and whether /device.data
                    device_feats[d_id, 1] = snr_db / self.device_sight_range  # SNR_db
                    device_feats[d_id, 2] = device.remaining_data / self.data_scale  # remaining data
                    # TODO: check whether the d_x/d_y is pose or index
                    device_feats[d_id, 3] = np.linalg.norm(pos - agent.current_pose) / self.dis_scale  # distance
                    device_feats[d_id, 4] = (self.pose_to_index(pos)[0, 0] - agent.current_pose_index[
                        0, 0]) / self.max_index_x  # relative x
                    device_feats[d_id, 5] = (self.pose_to_index(pos)[0, 1] - agent.current_pose_index[
                        0, 1]) / self.max_index_y  # relative y

            # Ally features
            a_ids = [a_id for a_id in range(self.n_agents) if a_id != agent_id]
            for i, a_id in enumerate(a_ids):
                ally = self.get_agent_by_id(a_id)
                snr_db = self.radio_ch_model.snr_measurement(self.city, ally.current_pose, agent.current_pose, 'uav')
                if snr_db >= self.uav_sight_range:
                    ally_feats[i, 0] = 1  # whether communicate with this UAV
                    ally_feats[i, 1] = snr_db / self.device_sight_range  # SNR_db
                    ally_feats[i, 2] = ally.battery / self.battery_scale  # remaining battery
                    ally_feats[i, 3] = np.linalg.norm(
                        ally.current_pose - agent.current_pose) / self.dis_scale  # distance
                    ally_feats[i, 4] = (ally.current_pose_index[0, 0] - agent.current_pose_index[
                        0, 0]) / self.max_index_x  # relative x
                    ally_feats[i, 5] = (ally.current_pose_index[0, 1] - agent.current_pose_index[
                        0, 1]) / self.max_index_y  # relative y

            # Own features
            own_feats[0] = agent.battery / self.battery_scale  # remaining battery
            _, extra_battery_to_end = self.safety_controller(agent)
            own_feats[1] = extra_battery_to_end / self.battery_scale  # extra battery to destination
            own_feats[2] = (agent.current_pose_index[0, 0] - self.pose_to_index(agent.end_pose)[0, 0]) / self.max_index_x  # relative x to destination
            own_feats[3] = (agent.current_pose_index[0, 1] - self.pose_to_index(agent.end_pose)[0, 1]) / self.max_index_y  # relative y to destination

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                device_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        return agent_obs

    def get_state(self):
        """ Return the global state.
        Note: Can not be used when execution
        """
        state_dict = self.get_state_dict()

        state = np.append(
            state_dict["allies"].flatten(), state_dict["devices"].flatten()
        )
        if "last_action" in state_dict:
            state = np.append(state, state_dict["last_action"].flatten())

        # state = state.astype(dtype=np.float32)

        return state

    def get_state_dict(self):
        """Returns the global state as a dictionary.

        - allies: numpy array containing agents and their attributes
        - devices: numpy array containing devices and their attributes
        - last_action: numpy array of previous actions for each agent

        NOTE: This function should not be used during decentralised execution.
        """

        # number of features in global state of each UAV and device
        nf_al = len(self.ally_state_sttr_names)
        nf_de = len(self.device_state_attr_name)

        ally_state = np.zeros((self.n_agents, nf_al))
        device_state = np.zeros((self.n_devices, nf_de))

        # TODO: whether to add the connecting state of each device
        for al_id, agent in enumerate(self.agents):
            _, extra_battery_to_end = self.safety_controller(agent)
            ally_state[al_id, 0] = agent.battery / self.battery_scale  # remaining battery
            ally_state[al_id, 1] = extra_battery_to_end / self.battery_scale
            # TODO: whether devide the self.max_x/y as a scalar
            ally_state[al_id, 2] = agent.current_pose_index[0, 0] / self.max_index_x  # x
            ally_state[al_id, 3] = agent.current_pose_index[0, 1] / self.max_index_y  # y
            ally_state[al_id, 4] = agent.done  # whether the UAV finish the task

        for de_id, device in enumerate(self.device_list.devices):
            # TODO: if data == 0, how?
            device_state[de_id, 0] = device.remaining_data / self.data_scale  # remaining data
            device_state[de_id, 1] = self.pose_to_index(device.position)[0, 0] / self.max_index_x  # x
            device_state[de_id, 2] = self.pose_to_index(device.position)[0, 1] / self.max_index_y  # y

        state = {"allies": ally_state, "devices": device_state}

        if self.state_last_action:
            state["last_action"] = self.last_action

        return state

    def get_obs_size(self):
        """ Returns the size of the observation """
        move_feats_dim = self.obs_move_feats_size
        n_devices, device_feats_dim = self.obs_device_feats_size
        n_allies, ally_feats_dim = self.obs_ally_feats_size
        own_feats_dim = self.obs_own_feats_size

        size = move_feats_dim + n_devices * device_feats_dim \
               + n_allies * ally_feats_dim + own_feats_dim

        return size

    def get_state_size(self):
        """ Return the size of the global state"""
        nf_al = len(self.ally_state_sttr_names)  # number of UAVs
        nf_de = len(self.device_state_attr_name)  # number of devices

        size = self.n_agents * nf_al + self.n_devices * nf_de

        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_extra_battery_to_end:
            size += self.n_agents

        return size

    # TODO: IMPLEMENT?
    def get_avail_actions(self):
        """ Returns the available actions of all agents in a list """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        agent = self.get_agent_by_id(agent_id)
        avail_actions = [0] * self.n_actions

        # if the UAV arrives destination or runs out of battery, it can only take no-op action
        if agent.done is True or agent.battery <= 0:
            avail_actions = [0] * (self.n_actions - 1) + [1]
            return avail_actions

        avail_movement_actions, _ = self.safety_controller(agent)
        for i, action in enumerate(avail_movement_actions):
            if action != 0:
                pos_index = agent.current_pose_index + agent.action_space[i]
                avail_actions[i] = self.check_uav_pos(pos_index)
        # if the agent arrives the terminal position, it can choose non-op action
        if np.array_equal(agent.current_pose_index, agent.pose_to_index(agent.end_pose)):
            avail_actions[-1] = 1

        assert (sum(avail_actions) > 0), "Agent {} cannot preform action".format(agent_id)

        return avail_actions

    def get_avail_device(self, agent_id):
        """ Return the connected device's index with the highest SNR_db"""

        agent = self.get_agent_by_id(agent_id)
        snr_db_list = [0] * self.n_devices
        device_access = [0] * self.n_devices
        if agent.battery > 0 and agent.done is False:
            for d_id, device in enumerate(self.device_list.devices):
                if not device.depleted:
                    pos = device.position
                    snr_db = self.radio_ch_model.snr_measurement(self.city, pos, agent.current_pose, 'device')
                    # the signal strength is enough to let the UAV get access to the info of the device
                    if snr_db >= self.device_sight_range:
                        snr_db_list[d_id] = snr_db
            # _, idx = self.device_list.get_best_data_rate(collected_meas)
            # if idx != -1:
            #     device_access[idx] = 1
            idx = np.argmax(snr_db_list)
            device_access[idx] = 0 if all(snr_db == 0 for snr_db in snr_db_list) else 1

        return device_access

    def get_avail_devices(self):
        """ Returns the index of connected devices of all agents in a list """
        total_device_access = []
        for agent_id in range(self.n_agents):
            device_access = self.get_avail_device(agent_id)
            total_device_access.append(device_access)
        return total_device_access

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def get_agent_by_id(self, a_id):
        """Get ally UAV by ID."""
        return self.agents[a_id]

    # def get_device_by_id(self, d_id):
    #     """ Get device by ID """
    #     return self.device_list.
    def get_avial_agent_actions(self, agent_id):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.n_actions,
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    # TODO: make the code more general
    def init_link_status(self):
        save_path = './Data/CityMap/city_link_status.npy'
        if not os.path.exists(save_path):
            status = self.radio_ch_model.init_city_link_status(self.city, self.step_size)
            np.save(save_path, status)
        status = np.load(save_path)
        self.radio_ch_model.link_status = status

        return status
