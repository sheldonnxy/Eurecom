import torch
import unet
from data_processing import process_input, process_output

def predict(model, masked_measurement, true_map, device):
    model.eval()
    with torch.no_grad():
        input_tensor = process_input(masked_measurement, true_map)
        input_tensor = input_tensor.to(device)
        
        map_pred, uncertainty = model(input_tensor)
        
        map_pred, uncertainty = process_output(map_pred, uncertainty)
    
    return map_pred, uncertainty

if __name__ == "__main__":
    # Load the trained model
    model = UNet(in_channels=2, out_channels=1)
    model.load_state_dict(torch.load("trained_unet_model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # masked_measurement = ...
    # true_map = ...
    
    predicted_map, predicted_uncertainty = predict(model, masked_measurement, true_map, device)
    