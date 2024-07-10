import numpy as np
import torch
import numpy as np
import gym
from gym import spaces

# setting the reward, including the uncertainty map
# use the ground truth radio map
# optimal trajectory can be used to improve the pre-trained UNet, as a rolling process

class UAVEnv(gym.Env):
    def __init__(self, radio_map, uncertainty_map, num_agents=3, steps=6):
        super(UAVEnv, self).__init__()
        self.radio_map = radio_map
        self.uncertainty_map = uncertainty_map
        self.num_agents = num_agents
        self.steps = steps
        
        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # Up, Down, Left, Right, Stay
        self.observation_space = spaces.Box(low=0, high=1, shape=(radio_map.shape[0], radio_map.shape[1], 3), dtype=np.float32)
        
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=np.int32)
        self.reset()

    def reset(self):
        # Reset agent positions and environment state
        self.agent_positions = np.random.randint(0, self.radio_map.shape[0], size=(self.num_agents, 2))
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        # Combine radio map, uncertainty map, and agent positions
        obs = np.stack([self.radio_map, self.uncertainty_map, np.zeros_like(self.radio_map)], axis=-1)
        for pos in self.agent_positions:
            obs[pos[0], pos[1], 2] = 1  # Mark agent positions
        return obs

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        for i, action in enumerate(actions):
            if action == 0:  # Up
                self.agent_positions[i][0] = max(0, self.agent_positions[i][0] - 1)
            elif action == 1:  # Down
                self.agent_positions[i][0] = min(self.radio_map.shape[0] - 1, self.agent_positions[i][0] + 1)
            elif action == 2:  # Left
                self.agent_positions[i][1] = max(0, self.agent_positions[i][1] - 1)
            elif action == 3:  # Right
                self.agent_positions[i][1] = min(self.radio_map.shape[1] - 1, self.agent_positions[i][1] + 1)
            # No movement for action == 4

            # Reward based on reduction in uncertainty
            rewards[i] = -self.uncertainty_map[self.agent_positions[i][0], self.agent_positions[i][1]]
            self.uncertainty_map[self.agent_positions[i][0], self.agent_positions[i][1]] *= 0.9  # Reduce uncertainty

        self.current_step += 1
        done = self.current_step >= self.steps
        return self._get_obs(), rewards, done, {}

class RadioMapEnvironment:
    def __init__(self, map_size, num_uavs, steps_per_episode):
        self.map_size = map_size
        self.num_uavs = num_uavs
        self.steps_per_episode = steps_per_episode
        self.reset()

    

    def calculate_reward(self, uav_index):
        x, y = self.uav_positions[uav_index]
        return 0
    
    
    
    
    # R1: Energy consumption (effective sampling ratio)
    new_data_count = np.sum(self.current_map > 0) - np.sum(self.current_map[x, y] > 0)
    es = new_data_count / self.step_count if self.step_count > 0 else 0
    r1 = es
    
    # R2: Uncertainty collection
    r2 = self.uncertainty_map[x, y]
    
    # R3: Accuracy of radio map estimation (NAE)
    if self.step_count == self.steps_per_episode - 1:
        nae = np.mean(np.abs(self.current_map - self.ground_truth_map))
        r3 = -nae
    else:
        r3 = 0
    
    # Combine rewards
    total_reward = r1 + r2 + r3
    return total_reward