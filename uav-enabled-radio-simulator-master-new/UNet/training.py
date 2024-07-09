import torch
import torch.optim as optim
from unet import UNet
from data_processing import process_input, process_output
from loss_functions import custom_loss

def train(model, train_loader, val_loader, num_epochs, learning_rate, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            masked_measurement, true_map, mask = batch
            input_tensor = process_input(masked_measurement, true_map)
            input_tensor = input_tensor.to(device)
            true_map = true_map.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            map_pred, uncertainty = model(input_tensor)
            
            loss, mse_loss, uncertainty_loss = custom_loss(map_pred, uncertainty, true_map, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                masked_measurement, true_map, mask = batch
                input_tensor = process_input(masked_measurement, true_map)
                input_tensor = input_tensor.to(device)
                true_map = true_map.to(device)
                mask = mask.to(device)
                
                map_pred, uncertainty = model(input_tensor)
                loss, _, _ = custom_loss(map_pred, uncertainty, true_map, mask)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
    
    return model

if __name__ == "__main__":
    # Set up data loaders, model, and training parameters here
    model = UNet(in_channels=2, out_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # train_loader = ...
    # val_loader = ...
    
    trained_model = train(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device=device)
    
    #Save the trained model
    torch.save(trained_model.state_dict(), "trained_unet_model.pth")