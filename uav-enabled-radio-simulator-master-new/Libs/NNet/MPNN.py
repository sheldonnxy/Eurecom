import torch
import torch.nn as nn
import torch.nn.functional as f


class MPNN(nn.Module):
    def __init__(self, hidden_layers, input_size, output_size):
        super(MPNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers[0][0])
        self.fc2 = nn.Linear(hidden_layers[0][0], hidden_layers[1][0])
        self.fc3 = nn.Linear(hidden_layers[1][0], output_size)


    def forward(self, inputs):
        x = torch.tanh(self.fc1(inputs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
