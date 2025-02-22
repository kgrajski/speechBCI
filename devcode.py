
#
# Refs:
# https://medium.com/correll-lab/a-deep-dive-into-autoencoders-ae-vae-and-vq-vae-with-code-ba712b9210eb
# https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
# https://medium.com/@mrunmayee.dhapre/using-variational-autoencoders-vae-for-time-series-data-reduction-9681338a2e17


import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(flat_input, self.embedding.weight.t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim//2, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=hidden_dim//2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.residual_stack = ResidualStack(hidden_dim, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=hidden_dim//2, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=hidden_dim//2, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.residual_stack = ResidualStack(hidden_dim, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        x = self.residual_stack(x)
        x = self.relu1(self.conv_transpose1(x))
        return self.conv_transpose2(x)

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, num_residual_hiddens):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return x + self.res_block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([ResidualLayer(in_channels, num_residual_hiddens) for _ in range(self.num_residual_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, num_residual_layers, num_residual_hiddens)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_dim, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss
    
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define hyperparameters
batch_size = 32
learning_rate = 1e-3
num_epochs = 10

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
model = VQVAE(in_channels=3, hidden_dim=128, num_residual_layers=2, num_residual_hiddens=32,
              num_embeddings=512, embedding_dim=64, commitment_cost=0.25)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
model.train()
for epoch in range(num_epochs):
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, _ = data
        optimizer.zero_grad()
        outputs, vq_loss = model(inputs)
        recon_error = torch.mean((outputs - inputs)**2)
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss / 100:.3f}')
            train_loss = 0.0

print('Finished Training')

# Test the model
model.eval()
test_loss = 0.0
with torch.no_grad():
    for data in test_loader:
        images, _ = data
        outputs, vq_loss = model(images)
        recon_error = torch.mean((outputs - images)**2)
        loss = recon_error + vq_loss
        test_loss += loss.item()

print(f'Test Loss: {test_loss / len(test_loader):.3f}')

import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(flat_input, self.embedding.weight.t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=hidden_dim//2, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=hidden_dim//2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.residual_stack = ResidualStack(hidden_dim, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.conv_transpose1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=hidden_dim//2, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose3d(in_channels=hidden_dim//2, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.residual_stack = ResidualStack(hidden_dim, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        x = self.residual_stack(x)
        x = self.relu1(self.conv_transpose1(x))
        return self.conv_transpose2(x)

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, num_residual_hiddens):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=num_residual_hiddens, out_channels=in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return x + self.res_block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([ResidualLayer(in_channels, num_residual_hiddens) for _ in range(self.num_residual_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, num_residual_layers, num_residual_hiddens)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_dim, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss
    
import torch.optim as optim

# Define hyperparameters
batch_size = 32
learning_rate = 1e-3
num_epochs = 10

# Define data transformations
# Define data transformations
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, 0  # Returning 0 as a dummy label

# Generate synthetic 3D data (replace with your actual data loading)
def generate_synthetic_3d_data(num_samples, depth, height, width, channels):
    return np.random.rand(num_samples, depth, height, width, channels).astype(np.float32)

# Define data transformations
class ToTensor3D:
    def __call__(self, sample):
        # Convert numpy array to tensor
        sample = torch.from_numpy(sample)
        # Reorder dimensions to (C, D, H, W)
        sample = sample.permute(3, 0, 1, 2)
        return sample

# Create a synthetic dataset
num_samples = 1000
depth = 16
height = 32
width = 32
channels = 3
synthetic_data = generate_synthetic_3d_data(num_samples, depth, height, width, channels)

# Define the transform
transform = transforms.Compose([
    ToTensor3D(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust normalization as needed
])

# Create the dataset
dataset = CustomDataset(synthetic_data, transform=transform)

# Split the dataset into training and testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
model = VQVAE(in_channels=3, hidden_dim=128, num_residual_layers=2, num_residual_hiddens=32,
              num_embeddings=512, embedding_dim=64, commitment_cost=0.25)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
model.train()
for epoch in range(num_epochs):
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, _ = data
        optimizer.zero_grad()
        outputs, vq_loss = model(inputs)
        recon_error = torch.mean((outputs - inputs)**2)
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss / 100:.3f}')
            train_loss = 0.0

print('Finished Training')

# Test the model
model.eval()
test_loss = 0.0
with torch.no_grad():
    for data in test_loader:
        images, _ = data
        outputs, vq_loss = model(images)
        recon_error = torch.mean((outputs - images)**2)
        loss = recon_error + vq_loss
        test_loss += loss.item()

print(f'Test Loss: {test_loss / len(test_loader):.3f}')

# Define a custom dataset for 3D data
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, 0  # Returning 0 as a dummy label

# Generate synthetic 3D data (replace with your actual data loading)
def generate_synthetic_3d_data(num_samples, depth, height, width, channels):
    return np.random.rand(num_samples, depth, height, width, channels).astype(np.float32)

# Define a transform to convert numpy array to tensor and reorder dimensions
class ToTensor3D:
    def __call__(self, sample):
        # Convert numpy array to tensor
        sample = torch.from_numpy(sample)
        # Reorder dimensions to (C, D, H, W)
        sample = sample.permute(3, 0, 1, 2)
        return sample

# Example usage:
if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 2

    # Generate synthetic data
    num_samples = 100
    depth = 8
    height = 16
    width = 16
    channels = 3
    synthetic_data = generate_synthetic_3d_data(num_samples, depth, height, width, channels)

    # Define the transform
    transform = transforms.Compose([
        ToTensor3D(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust normalization as needed
    ])

    # Create the dataset
    dataset = CustomDataset(synthetic_data, transform=transform)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    model = VQVAE(in_channels=3, hidden_dim=64, num_residual_layers=1, num_residual_hiddens=16,
                  num_embeddings=128, embedding_dim=32, commitment_cost=0.25)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, _ = data
            optimizer.zero_grad()
            outputs, vq_loss = model(inputs)
            recon_error = torch.mean((outputs - inputs)**2)
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        print(f'Epoch {epoch+1}, Average Loss: {train_loss / len(dataloader):.4f}')

    print('Finished Training')
    
    
    #
    #
    #
    #
    import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define the VQ-VAE model
class VQVAE(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_embeddings):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, loss_vq, perplexity = self.vq_layer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, loss_vq, perplexity

# Define the Vector Quantizer layer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost

    def forward(self, z_e):
        # Flatten input
        z_e_flat = z_e.view(-1, self.embedding_dim)

        # Calculate distances
        distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) \
                    + torch.sum(self.embedding.weight**2, dim=1) \
                    - 2 * torch.matmul(z_e_flat, self.embedding.weight.t())

        # Find closest embedding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z_e.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        z_q_flat = torch.matmul(encodings, self.embedding.weight)
        z_q = z_q_flat.view(z_e.shape)

        # Loss
        e_latent_loss = torch.mean((z_q.detach() - z_e)**2)
        q_latent_loss = torch.mean((z_q - z_e.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, loss, perplexity

# Create a dummy dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples, seq_len, input_dim):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.data = np.random.rand(num_samples, seq_len, input_dim).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

# Training loop
def train(model, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss, total_vq_loss = 0, 0
        for batch in dataloader:
            batch = batch.to(device)
            batch_size, seq_len, input_dim = batch.shape
            # Reshape to (batch_size * seq_len, input_dim)
            batch_reshaped = batch.view(-1, input_dim)
            optimizer.zero_grad()
            x_hat, loss_vq, _ = model(batch_reshaped)
            loss_reconstruction = nn.MSELoss()(x_hat, batch_reshaped)
            loss = loss_reconstruction + loss_vq
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_vq_loss += loss_vq.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}, VQ Loss: {total_vq_loss/len(dataloader)}")

# Testing loop
def test(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            batch_size, seq_len, input_dim = batch.shape
            # Reshape to (batch_size * seq_len, input_dim)
            batch_reshaped = batch.view(-1, input_dim)
            x_hat, _, _ = model(batch_reshaped)
            loss = nn.MSELoss()(x_hat, batch_reshaped)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss/len(dataloader)}")

# Set hyperparameters
input_dim = 3
embedding_dim = 2
num_embeddings = 5
epochs = 10
batch_size = 32
learning_rate = 1e-3
num_samples = 1000
seq_len = 20

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset and dataloaders
train_dataset = TimeSeriesDataset(num_samples, seq_len, input_dim)
test_dataset = TimeSeriesDataset(num_samples // 5, seq_len, input_dim)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate model and optimizer
model = VQVAE(input_dim, embedding_dim, num_embeddings).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train and test the model
train(model, train_dataloader, optimizer, epochs, device)
test(model, test_dataloader, device)




