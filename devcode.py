import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn as nn
import torch.optim as optim

# Define the VQ-VAE model (replace with your actual model)
class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        # Define layers (example)
        self.encoder = nn.Sequential(nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU())
        self.vq_layer = VectorQuantization(num_embeddings=10, embedding_dim=64*7*7) # Example values
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                                     nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, 64*7*7)
        vq_output = self.vq_layer(z_flattened)
        x_recon = self.decoder(z) #Decoder needs to take the output of the encoder
        return x_recon, vq_output.loss, vq_output.perplexity, vq_output.encodings

class VectorQuantization(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantization, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return VQVAEOutput(loss, quantized, perplexity, encodings, encoding_indices)

class VQVAEOutput:
    def __init__(self, loss, quantized, perplexity, encodings, encoding_indices):
        self.loss = loss
        self.quantized = quantized
        self.perplexity = perplexity
        self.encodings = encodings
        self.encoding_indices = encoding_indices

# Training parameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST Dataset (or replace with your dataset)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model, optimizer, and loss
model = VQVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)

        # Forward pass
        x_recon, vq_loss, perplexity, _ = model(images)
        recon_loss = nn.MSELoss()(x_recon, images) # Or another reconstruction loss
        loss = recon_loss + vq_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Recon Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Perplexity: {perplexity.item():.4f}')

# Testing loop
model.eval()
with torch.no_grad():
    test_loss = 0
    for images, _ in test_loader:
        images = images.to(device)
        x_recon, vq_loss, perplexity, _ = model(images)
        recon_loss = nn.MSELoss()(x_recon, images)
        loss = recon_loss + vq_loss
        test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')