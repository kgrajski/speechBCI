import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling block with maxpool and double convolution"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling block with skip connections"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Padding if needed for dimensions mismatch
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                       diffH // 2, diffH - diffH // 2,
                       diffD // 2, diffD - diffD // 2])
        
        # Concatenate along the channels dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[64, 128, 256, 512]):
        """
        3D U-Net for MRI segmentation
        Args:
            in_channels: number of input channels (1 for grayscale MRI)
            out_channels: number of output classes (e.g., 2 for binary segmentation)
            features: list of feature dimensions for each layer
        """
        super(UNet3D, self).__init__()
        
        # Encoder path
        self.encoder1 = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        
        # Decoder path
        self.up1 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        
        # Final convolution
        self.final_conv = nn.Conv3d(features[1], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        
        # Final convolution
        output = self.final_conv(x)
        
        return output

# Usage example
if __name__ == "__main__":
    # Create a dummy 3D MRI volume (batch_size=1, channels=1, depth=64, height=128, width=128)
    mri_volume = torch.randn(1, 1, 64, 128, 128)
    
    # Initialize model
    model = UNet3D(in_channels=1, out_channels=2)
    
    # Forward pass
    segmentation = model(mri_volume)
    
    print(f"Input shape: {mri_volume.shape}")
    print(f"Output shape: {segmentation.shape}")