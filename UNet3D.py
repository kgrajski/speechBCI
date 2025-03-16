import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib  # For loading MRI images
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Define the 3D U-Net architecture
class DoubleConv(nn.Module):
    """Double convolution block with batch normalization"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[64, 128, 256, 512]):
        super(UNet3D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Encoder path
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder path
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose3d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections list
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # Handle size mismatch (if any)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=True)
                
            # Concatenate
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)
            
        # Final output
        return self.final_conv(x)

# Custom MRI Dataset
class MRIDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images_list = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
        
    def __len__(self):
        return len(self.images_list)
        
    def __getitem__(self, idx):
        img_name = self.images_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        # Load MRI image and mask using nibabel
        image = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).long()
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            
        return image, mask

# Dice loss for segmentation
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs[:, 1].view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice

# Function to calculate Dice coefficient
def dice_coefficient(y_pred, y_true, smooth=1e-6):
    y_pred = torch.argmax(y_pred, dim=1)
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()
    best_model_wts = model.state_dict()
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_dice = 0.0
            
            # Iterate over data
            for inputs, masks in tqdm(dataloader):
                inputs = inputs.to(device)
                masks = masks.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    dice = dice_coefficient(outputs, masks)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_dice += dice.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_dice = running_dice / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Dice: {epoch_dice:.4f}')
            
            # Update learning rate
            if phase == 'train':
                scheduler.step()
                
            # Deep copy the model if best
            if phase == 'val' and epoch_dice > best_dice:
                best_dice = epoch_dice
                best_model_wts = model.state_dict().copy()
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': best_dice,
                }, 'best_model_unet3d.pth')
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Dice: {best_dice:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Testing function
def test_model(model, test_loader, device='cuda'):
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for inputs, masks in tqdm(test_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            outputs = model(inputs)
            dice = dice_coefficient(outputs, masks)
            dice_scores.append(dice.item())
    
    avg_dice = np.mean(dice_scores)
    print(f'Average Dice score: {avg_dice:.4f}')
    return avg_dice

def main():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Define transformations
    transform = transforms.Compose([
        # Add any transformations here - normalization, augmentations, etc.
    ])
    
    # Create datasets
    train_dataset = MRIDataset(
        images_dir='path/to/train/images',
        masks_dir='path/to/train/masks',
        transform=transform
    )
    
    val_dataset = MRIDataset(
        images_dir='path/to/val/images',
        masks_dir='path/to/val/masks',
        transform=transform
    )
    
    test_dataset = MRIDataset(
        images_dir='path/to/test/images',
        masks_dir='path/to/test/masks',
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Initialize model
    model = UNet3D(in_channels=1, out_channels=2)  # 2 channels for binary segmentation
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Train model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=25,
        device=device
    )
    
    # Test model
    test_model(model, test_loader, device=device)

if __name__ == "__main__":
    main()