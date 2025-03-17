# -*- coding: utf-8 -*-
"""Vqvae_Simple3D.py

Implementation of a Vector Quantized Variational Autoencoder (VQ-VAE) with 3D convolutional layers.

This module provides a simple implementation of the VQ-VAE architecture that uses 3D convolutional 
layers for processing volumetric data. The implementation includes vector quantization with both 
standard and EMA-based approaches.

ACKNOWLEDGEMENTS:
See Vqvae_Classic2D.py for the acknowledgements.

Classes:
    VectorQuantizer: A PyTorch module for vector quantization.
    VectorQuantizerEMA: A PyTorch module for vector quantization with exponential moving average.
    Encoder: A 3D convolutional encoder for feature extraction.
    PreVQLayer: A convolutional layer to adjust dimensionality before vector quantization.
    PostVQLayer: A transposed convolutional layer to adjust dimensionality after vector quantization.
    Decoder: A 3D convolutional decoder for reconstruction.
    VQVAE: The complete VQ-VAE model combining all components.

Usage example:
    model = VQVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.randn(16, 2, 32, 32, 32)  # Example input data (batch, channels, depth, height, width)
    loss, recon, perplexity = model(data)
    loss.backward()
    optimizer.step()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Vector Quantization layer as described in VQ-VAE paper.
    
    This layer takes a tensor to be quantized. The channel dimension is used as the space
    in which to quantize, while all other dimensions are flattened and treated as different
    examples to quantize.
    
    For 3D data with shape [B, C, D, H, W], the tensor is first converted to [B, D, H, W, C],
    then flattened to [B*D*H*W, C] for quantization, and finally reshaped back to the original format.
    
    Args:
        num_embeddings (int): Size of the codebook (number of embedding vectors). Default: 64
        embedding_dim (int): Dimension of each embedding vector. Default: 64
        commitment_cost (float): Weight for the commitment loss. Default: 0.25
        decay (float): Not used in this class, included for API compatibility. Default: 0.99
        epsilon (float): Not used in this class, included for API compatibility. Default: 1e-5
    """
    def __init__(self, num_embeddings=64, embedding_dim=64, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        """Perform vector quantization on the input tensor.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, D, H, W].
            
        Returns:
            tuple:
                - loss (torch.Tensor): The VQ loss (quantization + commitment).
                - quantized (torch.Tensor): The quantized representation, same shape as input.
                - perplexity (torch.Tensor): A measure of codebook usage.
                - encodings (torch.Tensor): One-hot encodings of shape [B*D*H*W, num_embeddings].
        """
        # convert inputs from BCDHW -> BDHWC
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape

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
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BDHWC -> BCDHW
        return loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), perplexity, encodings

class VectorQuantizerEMA(nn.Module):
    """Vector Quantization layer with Exponential Moving Average updates.
    
    Similar to VectorQuantizer but uses EMA to update the embedding vectors,
    which can lead to more stable training. The codebook is updated using an
    exponential moving average of the cluster assignments and embeddings.
    
    Args:
        num_embeddings (int): Size of the codebook (number of embedding vectors). Default: 64
        embedding_dim (int): Dimension of each embedding vector. Default: 64
        commitment_cost (float): Weight for the commitment loss. Default: 0.25
        decay (float): Decay rate for EMA updates. Default: 0.99
        epsilon (float): Small constant to avoid division by zero. Default: 1e-5
    """
    def __init__(self, num_embeddings=64, embedding_dim=64, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        """Perform vector quantization with EMA updates on the input tensor.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape [B, C, D, H, W].
            
        Returns:
            tuple:
                - loss (torch.Tensor): The commitment loss.
                - quantized (torch.Tensor): The quantized representation, same shape as input.
                - perplexity (torch.Tensor): A measure of codebook usage.
                - encodings (torch.Tensor): One-hot encodings of shape [B*D*H*W, num_embeddings].
        """
        # convert inputs from BCDHW ->  BDHWC
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape

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
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BDHWC -> BCDHW
        return loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), perplexity, encodings

class Encoder(nn.Module):
    """3D convolutional encoder for feature extraction.
    
    Consists of three convolutional blocks, each with batch normalization and ReLU activation.
    Progressively downsamples the input while increasing the number of channels.
    
    Args:
        in_channels (int): Number of input channels. Default: 2
        out_channels (int): Number of output channels. Default: 128
        kernel_size (int): Size of the convolving kernel. Default: 2
        stride (int): Stride of the convolution. Default: 2
        padding (int): Zero-padding added to all sides of the input. Default: 0
    """
    def __init__(self, in_channels=2, out_channels=128, kernel_size=2, stride=2, padding=0):
        super(Encoder, self).__init__()
        
        self._conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels//2,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm3d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        self._conv2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels//2,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self._conv3 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=2,
                      stride=2,
                      padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        """Encode the input tensor into a latent representation.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape [B, in_channels, D, H, W].
            
        Returns:
            torch.Tensor: Encoded representation of shape [B, out_channels, D', H', W'],
                          where dimensions are reduced due to strided convolutions.
        """
        x = self._conv1(inputs)
        x = self._conv2(x)
        x = self._conv3(x)
        return x
    
class PreVQLayer(nn.Module):
    """Convolutional layer to adjust feature dimensionality before vector quantization.
    
    This layer reduces the channel dimension to match the embedding dimension
    required by the vector quantizer.
    
    Args:
        in_channels (int): Number of input channels. Default: 128
        out_channels (int): Number of output channels (embedding dimension). Default: 64
        kernel_size (int): Size of the convolving kernel. Default: 2
        stride (int): Stride of the convolution. Default: 2
        padding (int): Zero-padding added to all sides of the input. Default: 0
    """
    def __init__(self, in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0):
        super(PreVQLayer, self).__init__()
        self._pre_vq_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=2, padding=0)

    def forward(self, inputs):
        """Adjust the channel dimension before vector quantization.
        
        Args:
            inputs (torch.Tensor): Input tensor from the encoder.
            
        Returns:
            torch.Tensor: Tensor with adjusted channel dimension for vector quantization.
        """
        x = self._pre_vq_conv(inputs)
        return x
    
class PostVQLayer(nn.Module):
    """Transposed convolutional layer to adjust feature dimensionality after vector quantization.
    
    This layer increases the channel dimension from the embedding dimension
    back to the dimension expected by the decoder.
    
    Args:
        in_channels (int): Number of input channels (embedding dimension). Default: 64
        out_channels (int): Number of output channels. Default: 128
        kernel_size (int): Size of the convolving kernel. Default: 2
        stride (int): Stride of the convolution. Default: 2
        padding (int): Zero-padding added to all sides of the input. Default: 0
    """
    def __init__(self, in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0):
        super(PostVQLayer, self).__init__()
        self._post_vq_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2, padding=0)

    def forward(self, inputs):
        """Adjust the channel dimension after vector quantization.
        
        Args:
            inputs (torch.Tensor): Quantized tensor from the vector quantizer.
            
        Returns:
            torch.Tensor: Tensor with channel dimension adjusted for the decoder.
        """
        x = self._post_vq_conv(inputs)
        return x

class Decoder(nn.Module):
    """3D convolutional decoder for reconstructing the input.
    
    Mirror structure to the encoder, using transposed convolutions to upsample
    the latent representation back to the original input dimensions.
    
    Args:
        in_channels (int): Number of input channels. Default: 128
        out_channels (int): Number of output channels (same as original input). Default: 2
        kernel_size (int): Size of the convolving kernel. Default: 2
        stride (int): Stride of the convolution. Default: 2
        padding (int): Zero-padding added to all sides of the input. Default: 0
    """
    def __init__(self, in_channels=128, out_channels=2, kernel_size=2, stride=2, padding=0):
        super(Decoder, self).__init__()
        
        self._convt1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=2,
                               stride=2,
                               padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self._convt2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels,
                      out_channels=in_channels//2,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm3d(in_channels//2),
            nn.ReLU(inplace=True)
        )

        self._convt3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels//2,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        """Reconstruct the input from the latent representation.
        
        Args:
            inputs (torch.Tensor): Input tensor from the post-VQ layer.
            
        Returns:
            torch.Tensor: Reconstructed tensor with the same spatial dimensions as the original input.
        """
        x = self._convt1(inputs)
        x = self._convt2(x)
        x = self._convt3(x)
        return x

class VQVAE(nn.Module):
    """Vector Quantized Variational Autoencoder with 3D convolutions.
    
    Combines an encoder, vector quantizer, and decoder to form a complete VQ-VAE model.
    The model takes volumetric data as input, compresses it into a discrete latent
    representation, and then reconstructs the input from this representation.
    
    The architecture flow is:
    input → encoder → pre_vq → vector_quantizer → post_vq → decoder → output
    """
    def __init__(self):
        super(VQVAE, self).__init__()
        self._encoder = Encoder()
        self._pre_vq = PreVQLayer()
        
            # Note that for now are using plain vanilla VectorQuantizer
        self._vq_vae = VectorQuantizer()
        
        self._post_vq = PostVQLayer()
        self._decoder = Decoder()

    def forward(self, x):
        """Forward pass through the entire VQ-VAE model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, D, H, W].
            
        Returns:
            tuple:
                - loss (torch.Tensor): The VQ loss.
                - x_recon (torch.Tensor): The reconstructed output tensor.
                - perplexity (torch.Tensor): A measure of codebook usage.
        """
        z = self._encoder(x)
        z = self._pre_vq(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._post_vq(quantized)
        x_recon = self._decoder(x_recon)
        return loss, x_recon, perplexity
