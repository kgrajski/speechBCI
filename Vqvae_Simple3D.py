# -*- coding: utf-8 -*-
"""Vqvae_Simple3D.py

Implementation of a Vector Quantized Variational Autoencoder (VQ-VAE) with 3D convolutional layers.

This module provides a comprehensive implementation of the VQ-VAE architecture optimized for
processing 3D volumetric neural data. The architecture includes 3D convolutional layers for
spatial feature extraction and vector quantization with both standard and EMA-based codebook updates.

The model is designed to compress 3D neural signals with spatial dimensions of [B, C, D, H, W]
into discrete latent representations, then reconstruct the original input from these representations.

ACKNOWLEDGEMENTS:
See Vqvae_Classic2D.py for the acknowledgements.

Classes:
    VectorQuantizer: Standard vector quantization with straight-through estimator.
    VectorQuantizerEMA: Vector quantization with exponential moving average codebook updates.
    Encoder: Progressive 3D convolutional encoder for feature extraction.
    PreVQLayer: Convolutional layer to prepare features for vector quantization.
    PostVQLayer: Convolutional layer to adapt quantized features for decoding.
    Decoder: Progressive 3D transposed convolutional decoder for reconstruction.
    VQVAE: Complete end-to-end VQ-VAE model combining all components.

Usage example:
    model = VQVAE(num_input_channels=4, num_output_channels=128, embedding_dim=512, num_embeddings=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.randn(16, 4, 8, 16, 8)  # Example neural data (batch, channels, depth, height, width)
    loss, recon, perplexity = model(data)
    loss.backward()
    optimizer.step()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Vector quantization layer implementing discrete codebook lookup.
    
    This layer maps continuous input vectors to discrete codes from a learned codebook.
    It rearranges input tensors to isolate the feature dimension for quantization,
    then maps each feature vector to the nearest codebook vector.
    
    For 3D neural data with shape [B, C, D, H, W], the layer:
    1. Permutes dimensions to [B, D, H, W, C]
    2. Flattens spatial dimensions to [B*D*H*W, C]
    3. Quantizes each vector by finding the nearest codebook entry
    4. Reshapes back to the original format
    
    The implementation uses a straight-through estimator for backpropagation through
    the non-differentiable quantization operation.
    
    Args:
        embedding_dim (int): Dimension of each codebook vector
        num_embeddings (int): Size of the codebook (number of discrete codes)
        commitment_cost (float): Weight for the commitment loss term (default: 0.25)
        decay (float): Unused parameter for API compatibility (default: 0.99)
        epsilon (float): Unused parameter for API compatibility (default: 1e-5)
    """

    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
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
        distances = (
            torch.sum(
                flat_input**2,
                dim=1,
                keepdim=True,
            )
            + torch.sum(
                self._embedding.weight**2,
                dim=1,
            )
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
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
        return (
            loss,
            quantized.permute(0, 4, 1, 2, 3).contiguous(),
            perplexity,
            encodings,
        )


class VectorQuantizerEMA(nn.Module):
    """Vector quantization layer with Exponential Moving Average codebook updates.
    
    This implementation achieves more stable training by updating the codebook
    using EMA rather than gradient descent. The codebook vectors are updated based on
    an exponential moving average of assigned input vectors.
    
    This approach helps prevent codebook collapse (where many codes become unused)
    and provides smoother convergence, especially with limited training data.
    
    The implementation includes Laplace smoothing of cluster sizes to prevent
    division by zero for rarely-used codebook vectors.
    
    Args:
        embedding_dim (int): Dimension of each codebook vector
        num_embeddings (int): Size of the codebook (number of discrete codes)
        commitment_cost (float): Weight for the commitment loss term (default: 0.25)
        decay (float): EMA decay rate for codebook updates (default: 0.99)
        epsilon (float): Small constant for numerical stability (default: 1e-5)
    """

    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
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
        distances = (
            torch.sum(
                flat_input**2,
                dim=1,
                keepdim=True,
            )
            + torch.sum(
                self._embedding.weight**2,
                dim=1,
            )
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BDHWC -> BCDHW
        return (
            loss,
            quantized.permute(0, 4, 1, 2, 3).contiguous(),
            perplexity,
            encodings,
        )


class Encoder(nn.Module):
    """Progressive 3D convolutional encoder for neural signal feature extraction.
    
    This encoder progressively reduces spatial dimensions while increasing the channel count
    through a series of strided 3D convolutions with BatchNorm and LeakyReLU activations.
    
    The architecture is designed specifically for neural data with initial dimensions
    [B, C, D, H, W] where typically C=4, D=8, H=16, W=8. The output dimensions are 
    [B, out_channels, D//8, H//8, W//8].
    
    The encoder includes four convolutional stages:
    1. First stage: Reduces spatial dimensions by factor of 2, increases channels to out_channels//4
    2. Second stage: Further reduces spatial dimensions, increases channels to out_channels//2
    3. Third stage: Final spatial reduction, increases channels to out_channels
    4. Fourth stage: Adjusts height dimension only, maintaining channel count
    
    Args:
        in_channels (int): Number of input channels (e.g., 4 for neural data)
        out_channels (int): Number of output channels (typically 128 or higher)
    """

    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self._conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels // 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm3d(out_channels // 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self._conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=out_channels // 4,
                out_channels=out_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm3d(out_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self._conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self._conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 2, 1),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
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
        x = self._conv4(x)
        return x


class PreVQLayer(nn.Module):
    """Pre-quantization layer to adapt encoder features for vector quantization.
    
    This layer uses a 1×1×1 convolution to adjust the channel dimension from the
    encoder output to match the required embedding dimension for vector quantization.
    It acts as a learned projection that maps the high-dimensional feature space
    to a space better suited for quantization.
    
    The spatial dimensions are preserved, while channel count is adjusted.
    BatchNorm and LeakyReLU help normalize and add non-linearity to the projection.
    
    Args:
        in_channels (int): Number of input channels from encoder (typically 128)
        embedding_dim (int): Target dimension for vector quantization (typically 512)
    """

    def __init__(self, in_channels, embedding_dim):
        super(PreVQLayer, self).__init__()
        self._pre_vq_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=embedding_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm3d(embedding_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

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
    """Post-quantization layer to adapt quantized vectors for decoding.
    
    After vector quantization, this layer converts the discrete codebook vectors
    back to the feature space expected by the decoder. It uses a 1×1×1 transposed
    convolution to expand the embedding dimension back to the decoder's expected
    input dimension.
    
    This layer provides flexibility in designing asymmetric encoder-decoder architectures
    where the bottleneck dimensions differ from the decoder's working dimensions.
    
    Args:
        embedding_dim (int): Dimension of quantized vectors (typically 512)
        out_channels (int): Number of output channels for decoder (typically 128)
    """

    def __init__(self, embedding_dim, out_channels):
        super(PostVQLayer, self).__init__()
        
        self._post_vq_conv = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=embedding_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm3d(out_channels),  # Changed from embedding_dim!
            nn.LeakyReLU(0.2, inplace=True),
        )

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
    """Progressive 3D transposed convolutional decoder for signal reconstruction.
    
    This decoder mirrors the encoder structure, using transposed convolutions to
    progressively upsample the spatial dimensions while decreasing the channel count.
    The goal is to accurately reconstruct the original input from the quantized
    latent representation.
    
    The decoder includes four transposed convolutional stages:
    1. First stage: Expands height dimension only, preserving channel count
    2. Second stage: Expands spatial dimensions by factor of 2, reduces channels to in_channels//2
    3. Third stage: Further expands spatial dimensions, reduces channels to in_channels//4
    4. Fourth stage: Final expansion to original dimensions, reduces channels to out_channels
       with Tanh activation to bound output values
    
    Args:
        in_channels (int): Number of input channels from post-VQ layer (typically 128)
        out_channels (int): Number of output channels for reconstruction (typically 4)
    """

    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self._convt1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 2, 1),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self._convt2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm3d(in_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self._convt3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm3d(in_channels // 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self._convt4 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels // 4,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
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
        x = self._convt4(x)
        return x


class VQVAE(nn.Module):
    """Complete Vector Quantized Variational Autoencoder for 3D neural data.
    
    This class integrates all components into a comprehensive end-to-end model:
    1. Encoder: Compresses spatial dimensions and extracts features
    2. PreVQ: Adjusts feature dimensions for quantization
    3. VQ: Performs vector quantization with codebook learning
    4. PostVQ: Adjusts quantized vectors for decoding
    5. Decoder: Reconstructs the original input
    
    The model is designed for neural signal processing, particularly for brain-computer
    interface applications with 3D electrode array recordings.
    
    Architecture flow:
    input → encoder → pre_vq → vector_quantizer → post_vq → decoder → output
    
    Args:
        num_input_channels (int): Number of input data channels (typically 4)
        num_output_channels (int): Number of encoder/decoder hidden channels (typically 128)
        embedding_dim (int): Dimension of codebook vectors (typically 512)
        num_embeddings (int): Number of discrete codes in codebook (typically 512)
    """

    def __init__(
        self,
        num_input_channels, # Refers to the input data
        num_output_channels, # Refers to the encoder output
        embedding_dim, # Refers to the VQ input and output dimensions
        num_embeddings, # Refers to the number of embeddings in the codebook
    ):
        super(VQVAE, self).__init__()
        self._encoder = Encoder(num_input_channels, num_output_channels)
        self._pre_vq = PreVQLayer(num_output_channels, embedding_dim)

        self._vq_vae = VectorQuantizer(embedding_dim, num_embeddings)

        self._post_vq = PostVQLayer(embedding_dim, num_output_channels)
        self._decoder = Decoder(num_output_channels, num_input_channels)

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
