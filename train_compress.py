import torch
import torch.nn as nn

class Compressor(nn.Module):
    def __init__(self, input_dim=256, time_steps=32, output_dim=512):
        super(Compressor, self).__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.output_dim = output_dim

        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(input_dim * time_steps, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)

        self.residual1 = nn.Linear(input_dim * time_steps, 1024)
        self.residual2 = nn.Linear(1024, 512)

        self.activation = nn.ReLU()

    def forward(self, x):
        # Flatten time steps into a single dimension
        x = x.view(x.size(0), -1)

        # First layer with residual connection
        residual = self.residual1(x)
        x = self.fc1(x)
        x = self.activation(x + residual)

        # Second layer with residual connection
        residual = self.residual2(x)
        x = self.fc2(x)
        x = self.activation(x + residual)

        # Final layer
        x = self.fc3(x)
        return x

# ...existing code for training loop...
# Replace the instantiation of the previous model with the new Compressor
model = Compressor(input_dim=256, time_steps=32, output_dim=512).to(device)

# ...existing code for optimizer, loss function, and training...
