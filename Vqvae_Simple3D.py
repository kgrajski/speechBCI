# -*- coding: utf-8 -*-
"""Vqvae_Simple3D.py

ACKNOWLEDGEMENTS:
See Vqvae_Classic2D.py for the acknowledgements.

This code implements a very simple VQ-VAE architecture that uses Conv3D layers.

Classes:
    VectorQuantizer: A PyTorch module for vector quantization.
    VectorQuantizerEMA: A PyTorch module for vector quantization with exponential moving average.
    Encoder: A simple 3D convolutional encoder.
    Decoder: A simple 3D convolutional decoder.
    VQVAE: A VQ-VAE model with 3D convolutional layers.

Usage example:
    model = VQVAE(encoder_in_channels=1, encoder_out_channels=64, kernel_size=3, stride=2, padding=1,
                  num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.randn(16, 1, 32, 32, 32)  # Example input data
    loss, recon, perplexity = model(data)
    loss.backward()
    optimizer.step()
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from six.moves import xrange

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

"""## Vector Quantizer Layer

This layer takes a tensor to be quantized.
The channel dimension will be used as the space in which to quantize.
All other dimensions will be flattened and will be seen as different
examples to quantize.

The output tensor will have the same shape as the input.

As an example for a `BCHW` tensor of shape `[16, 64, 32, 32]`,
we will first convert it to an `BHWC` tensor of shape `[16, 32, 32, 64]` and
then reshape it into `[16384, 64]` and all `16384` vectors of size `64` will be
quantized independently. In otherwords, the channels are used as the space
in which to quantize. All other dimensions will be flattened and be seen as
different examples to quantize, `16384` in this case.
"""
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
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

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), perplexity, encodings

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
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
        # 2D: convert inputs from BCHW -> BHWC
        # 2D: inputs = inputs.permute(0, 2, 3, 1).contiguous()
        # 3D: convert inputs from BCDHW ->  BDHWC
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

        # 2D: convert quantized from BHWC -> BCHW
        # 2D: return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        # 3D: convert quantized from BDHWC -> BCDHW
        #return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        return loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), perplexity, encodings
    
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, kernel_size, stride, padding, num_residual_hiddens):
            # Note the adjustment to kernel_size in the first Conv3d layer
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv3d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=kernel_size-1, stride=stride, padding=padding, bias=False),
            nn.ReLU(False),
            nn.Conv3d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, kernel_size, stride, padding,
                 num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens,
                                               kernel_size, stride, padding, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x, inplace=False)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 num_residual_layers, num_residual_channels):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv3d(in_channels=in_channels,
                                 out_channels=out_channels//2,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding)
        
        self._conv_2 = nn.Conv3d(in_channels=out_channels//2,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding)

            # Note: the residual stack in and out channels need to be the same
        self._residual_stack = ResidualStack(out_channels, out_channels,
                                             kernel_size, stride, padding,
                                             num_residual_layers, num_residual_channels)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x, inplace=False)
        x = self._conv_2(x)
        x = F.relu(x, inplace=False)
        x = self._residual_stack(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, encoder_out_channels, encoder_in_channels, kernel_size, stride, padding):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv3d(in_channels=embedding_dim, out_channels=encoder_out_channels,
                                 kernel_size=1, stride=1, padding=0)
        
        self._conv_trans_1 = nn.ConvTranspose3d(in_channels=encoder_out_channels,
                                                out_channels=encoder_out_channels//2,
                                                kernel_size=kernel_size, stride=stride, padding=padding)
        
        self._conv_trans_2 = nn.ConvTranspose3d(in_channels=encoder_out_channels//2,
                                                out_channels=encoder_in_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        x = self._conv_trans_2(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, encoder_in_channels, encoder_out_channels, kernel_size, stride, padding,
                 num_resid_layers, num_resid_channels, num_embeddings, embedding_dim, commitment_cost, decay):
        super(VQVAE, self).__init__()

        self._encoder = Encoder(encoder_in_channels, encoder_out_channels, kernel_size, stride, padding,
                                num_resid_layers, num_resid_channels)
        
        self._pre_vq_conv = nn.Conv3d(in_channels=encoder_out_channels, out_channels=embedding_dim,
                                      kernel_size=1, stride=1, padding=0)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        self._decoder = Decoder(embedding_dim, encoder_out_channels, encoder_in_channels, kernel_size, stride, padding)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity
