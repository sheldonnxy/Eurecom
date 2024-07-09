import torch
import torch.nn.functional as F

def custom_loss(map_pred, uncertainty, true_map, mask, alpha=0.5):
    # MSE loss for map prediction
    mse_loss = F.mse_loss(map_pred * mask, true_map * mask)
    
    # Custom loss for uncertainty
    delta = torch.abs(true_map - map_pred)
    uncertainty_loss = torch.mean(torch.square(delta - uncertainty) * mask)
    
    # Combine losses
    total_loss = alpha * mse_loss + (1 - alpha) * uncertainty_loss
    
    return total_loss, mse_loss, uncertainty_loss