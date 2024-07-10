import torch
import numpy as np
from environment import RadioMapEnvironment
from agent import MultiAgenti2q
from UNet import unet

def train(num_episodes, batch_size, map_size, num_uavs, steps_per_episode):
    env = RadioMapEnvironment(map_size, num_uavs, steps_per_episode)
    agents = MultiAgentDDPG(num_uavs, state_dim=map_size*map_size*2 + 2, action_dim=2)
    unet = UNet(n_channels=1, n_classes=1)
    replay_buffer = ReplayBuffer(1000000)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(steps_per_episode):
            # Preprocess state for agents
            flattened_state = np.concatenate([
                state['current_map'].flatten(),
                state['uncertainty_map'].flatten(),
                state['uav_positions'].flatten()
            ])
            agent_states = [flattened_state for _ in range(num_uavs)]

            # Select actions
            actions = agents.select_actions(agent_states)

            # Take actions in environment
            next_state, rewards, done = env.step(actions)

            # Store transition in replay buffer
            replay_buffer.add(state, actions, rewards, next_state, done)

            # Update agents
            if len(replay_buffer) > batch_size:
                agents.update(replay_buffer, batch_size)

            state = next_state
            episode_reward += sum(rewards)

            if done:
                break

        # Update UNet and uncertainty map
        with torch.no_grad():
            current_map_tensor = torch.FloatTensor(state['current_map']).unsqueeze(0).unsqueeze(0)
            uncertainty_map = unet(current_map_tensor).squeeze().numpy()
            env.uncertainty_map = uncertainty_map

        print(f"Episode {episode+1}, Reward: {episode_reward}")

    return agents, unet

class ReplayBuffer:
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = (state, action, reward, next_state, done)
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append((state, action, reward, next_state, done)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in ind:
            s, a, r, s_, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(s_, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(next_states), np.array(dones).reshape(-1, 1)

    def __len__(self):
        return len(self.storage)