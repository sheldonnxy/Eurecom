import torch
import numpy as np

def process_input(masked_measurement, true_map):
    # Concatenate masked measurement and true map
    input_data = np.concatenate([masked_measurement, true_map], axis=-1)
    
    # Convert to PyTorch tensor and add batch dimension
    input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)
    
    # Permute dimensions to [batch, channels, height, width]
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    
    return input_tensor

def process_output(map_pred, uncertainty):
    # Convert PyTorch tensors to numpy arrays
    map_pred = map_pred.squeeze().detach().cpu().numpy()
    uncertainty = uncertainty.squeeze().detach().cpu().numpy()
    
    return map_pred, uncertainty