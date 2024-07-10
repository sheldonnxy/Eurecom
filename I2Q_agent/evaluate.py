import numpy as np
import matplotlib.pyplot as plt
import torch
from UNet import *
unet_model = UNet()
unet_model.load_state_dict(torch.load('unet_model.pth'))


def predict_trajectories(agents, env, steps):
    trajectories = [[] for _ in range(env.num_uavs)]
    state = env.reset()

    for _ in range(steps):
        flattened_state = np.concatenate([
            state['current_map'].flatten(),
            state['uncertainty_map'].flatten(),
            state['uav_positions'].flatten()
        ])
        agent_states = [flattened_state for _ in range(env.num_uavs)]
        actions = agents.select_actions(agent_states)

        for i, action in enumerate(actions):
            trajectories[i].append(env.uav_positions[i])

        state, _, done = env.step(actions)
        if done:
            break

    return trajectories

def evaluate_radio_map(predicted_map, ground_truth_map):
    mse = np.mean((predicted_map - ground_truth_map) ** 2)
    mae = np.mean(np.abs(predicted_map - ground_truth_map))
    return mse, mae

def visualize_results(ground_truth_map, predicted_map, trajectories):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(ground_truth_map, cmap='viridis')
    ax1.set_title('Ground Truth Radio Map')

    ax2.imshow(predicted_map, cmap='viridis')
    ax2.set_title('Predicted Radio Map')

    ax3.imshow(ground_truth_map, cmap='viridis', alpha=0.5)
    for trajectory in trajectories:
        trajectory = np.array(trajectory)
        ax3.plot(trajectory )