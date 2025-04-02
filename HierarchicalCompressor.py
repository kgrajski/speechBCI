"""
Hierarchical Attention Compressor for neural time series data.

This module implements a hierarchical attention-based architecture for compressing
neural time series data into fixed-length representations suitable for language modeling.

References:
- Vaswani et al. "Attention Is All You Need" (2017)
- Yang et al. "Hierarchical Attention Networks for Document Classification" (2016)
- Child et al. "Generating Long Sequences with Sparse Transformers" (2019)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalAttentionCompressor(nn.Module):
    """
    Hierarchical attention-based compressor for neural time series.
    
    This model processes inputs in a hierarchical manner:
    1. Spatial compression: For each time window, process spatial positions
    2. Temporal compression: Process the sequence of time windows
    3. Fixed token extraction: Extract fixed number of tokens using learned queries
    """
    
    def __init__(
        self,
        input_dim=256,           # Dimension of input features per spatial position
        hidden_dim=256,          # Hidden dimension throughout the model
        output_dim=512,          # Output dimension of final tokens
        output_tokens=512,       # Number of output tokens
        spatial_shape=[8, 4],    # Height and width of spatial grid
        max_input_windows=250,   # Maximum number of time windows to process
        num_layers=4,            # Number of transformer layers
        num_heads=8,             # Number of attention heads
        ffn_dim=None,            # Dimension of feed-forward layers (default 4*hidden_dim)
        dropout=0.1              # Dropout rate
    ):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_tokens = output_tokens
        self.spatial_shape = spatial_shape
        self.spatial_tokens = spatial_shape[0] * spatial_shape[1]
        self.max_input_windows = max_input_windows
        
        # Feed-forward dimension defaults to 4x hidden_dim if not specified
        if ffn_dim is None:
            ffn_dim = 4 * hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Spatial components
        self.spatial_summary = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.spatial_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm1 = nn.LayerNorm(hidden_dim)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.spatial_norm2 = nn.LayerNorm(hidden_dim)
        
        # Temporal transformer
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )
        
        # Target queries for extracting fixed number of tokens
        self.target_queries = nn.Parameter(torch.randn(output_tokens, hidden_dim))
        self.query_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def process_spatial_window(self, window_tokens):
        """
        Process a single time window using spatial attention.
        
        Args:
            window_tokens: Tensor of shape [batch, spatial_tokens, hidden_dim]
                representing a single time window
        
        Returns:
            Tensor of shape [batch, hidden_dim] summarizing the time window
        """
        batch_size = window_tokens.size(0)
        
        # Expand summary token to batch size
        summary = self.spatial_summary.expand(batch_size, -1, -1)
        
        # Concatenate summary token with spatial tokens
        combined = torch.cat([summary, window_tokens], dim=1)
        
        # Apply self-attention where summary token attends to all spatial positions
        attended, _ = self.spatial_attention(
            summary,                   # Query (just the summary token)
            combined,                  # Key (summary + all spatial tokens)
            combined,                  # Value (summary + all spatial tokens)
            need_weights=False
        )
        
        # Residual connection and layer norm
        summary = summary + attended
        summary = self.spatial_norm1(summary)
        
        # Feed-forward network
        ffn_output = self.spatial_ffn(summary)
        summary = summary + ffn_output
        summary = self.spatial_norm2(summary)
        
        return summary.squeeze(1)
    
    def prepare_input_sequence(self, x):
        """
        Prepare the input sequence for processing.
        
        Args:
            x: Input tensor of shape [batch, time_windows, features]
               where features = channels*height*width (C*H*W = 256*8*4 = 8192)
        
        Returns:
            tuple: (processed_tensor, num_windows)
                - processed_tensor: Tensor of shape [batch, num_windows, spatial_tokens, hidden_dim]
                - num_windows: Number of time windows being processed
        """
        batch_size = x.size(0)
        time_windows = x.size(1)
        features = x.size(2)
        
        # Verify dimensions
        if features != self.spatial_tokens * self.input_dim:
            print(f"WARNING: Input features {features} don't match expected {self.spatial_tokens * self.input_dim}")
            # This would mean the spatial dimensions don't match what's expected
        
        # Limit to max input windows
        num_windows = min(time_windows, self.max_input_windows)
        
        # Truncate to max windows
        x = x[:, :num_windows]
        
        # Reshape to [batch, num_windows, spatial_tokens, input_dim]
        x = x.reshape(batch_size, num_windows, self.spatial_tokens, self.input_dim)
        
        # Project to hidden dimension
        x = self.input_projection(x.reshape(-1, self.input_dim))
        x = x.reshape(batch_size, num_windows, self.spatial_tokens, self.hidden_dim)
        
        return x, num_windows
    
    def forward(self, x, with_reconstruction=False):
        """
        Forward pass with optional reconstruction.
        
        Args:
            x: Input tensor of shape [batch, time_windows, features]
               or [batch, time_windows, channels, height, width]
               where features = C*H*W (typically 8192 for C=256, H=8, W=4)
            with_reconstruction: If True, also return reconstructed time windows
        
        Returns:
            If with_reconstruction is False:
                Tensor of shape [batch, output_tokens, output_dim]
            If with_reconstruction is True:
                Tuple of (compressed_tokens, reconstructed_windows, num_windows)
                - compressed_tokens: [batch, output_tokens, output_dim]
                - reconstructed_windows: [batch, num_windows, input_dim]
                - num_windows: Number of time windows processed
        """
        # Check and reshape input if it's 5D
        if len(x.shape) == 5:
            batch_size, time_windows, channels, height, width = x.shape
            x = x.reshape(batch_size, time_windows, channels * height * width)
        
        batch_size = x.size(0)
        
        # Prepare input sequence
        x_prepared, num_windows = self.prepare_input_sequence(x)
        
        # Handle empty sequence case
        if num_windows == 0:
            empty_tokens = torch.zeros(
                batch_size, self.output_tokens, self.output_dim, 
                device=x.device, dtype=x.dtype
            )
            if with_reconstruction:
                empty_windows = torch.zeros(
                    batch_size, 0, self.input_dim,
                    device=x.device, dtype=x.dtype
                )
                return empty_tokens, empty_windows, 0
            return empty_tokens
        
        # Process each time window to get summaries
        time_summaries = []
        for t in range(num_windows):
            window_summary = self.process_spatial_window(x_prepared[:, t])
            time_summaries.append(window_summary)
        
        # Stack window summaries
        time_sequence = torch.stack(time_summaries, dim=1)  # [batch, num_windows, hidden_dim]
        
        # Process with temporal transformer
        temporal_features = self.temporal_transformer(time_sequence)
        
        # Calculate attention scores for output token selection
        queries = self.query_norm(self.target_queries)  # [output_tokens, hidden_dim]
        scores = torch.einsum('td,bnd->btn', queries, temporal_features)
        scores = scores / math.sqrt(self.hidden_dim)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Compute weighted sum to get output tokens
        output_tokens = torch.einsum('btn,bnd->btd', attn_weights, temporal_features)
        
        # Project to output dimension
        output_tokens = self.output_projection(output_tokens)
        
        # Return with or without reconstruction
        if with_reconstruction:
            return output_tokens, temporal_features, num_windows
        
        return output_tokens


class HierarchicalCompressorWithReconstruction(HierarchicalAttentionCompressor):
    """
    Extension of the hierarchical compressor with reconstruction capability.
    
    This model adds a reconstruction pathway that can recover the original
    time windows from the compressed representation.
    """
    
    def __init__(self, **kwargs):
        """Initialize with the same parameters as the base compressor."""
        super().__init__(**kwargs)
        
        # Create reconstruction components
        self.recon_projection = nn.Linear(self.hidden_dim, self.input_dim)
    
    def reconstruct(self, compressed_tokens, num_windows):
        """
        Reconstruct original time window representations from compressed tokens.
        
        Args:
            compressed_tokens: Tensor of shape [batch, output_tokens, hidden_dim]
            num_windows: Number of time windows to reconstruct
        
        Returns:
            Tensor of shape [batch, num_windows, input_dim] representing the 
            channel dimension only (C=256) of the original input
        """
        batch_size = compressed_tokens.size(0)
        
        # Handle empty case
        if num_windows == 0:
            return torch.zeros(
                batch_size, 0, self.input_dim,
                device=compressed_tokens.device, dtype=compressed_tokens.dtype
            )
        
        # Project back to the original channel dimension using the hidden dim vectors
        reconstructed = self.recon_projection(compressed_tokens)
        
        # Take average over tokens to get time window representations
        # This is a simple approach; a more complex approach would use attention
        reconstructed = reconstructed.mean(dim=1, keepdim=True)
        reconstructed = reconstructed.expand(-1, num_windows, -1)
        
        return reconstructed
    
    def forward(self, x, with_reconstruction=False):
        """
        Forward pass with optional reconstruction.
        
        Args:
            x: Input tensor of shape [batch, time_windows, features]
            with_reconstruction: If True, also return reconstructed time windows
        
        Returns:
            If with_reconstruction is False:
                Tensor of shape [batch, output_tokens, output_dim]
            If with_reconstruction is True:
                Tuple of (compressed_tokens, reconstructed_windows, num_windows)
        """
        # Get compressed representation from parent class
        compressed_tokens, temporal_features, num_windows = super().forward(
            x, with_reconstruction=True
        )
        
        if with_reconstruction:
            # Apply reconstruction to get back original time windows
            reconstructed = self.reconstruct(compressed_tokens, num_windows)
            return compressed_tokens, reconstructed, num_windows
        
        return compressed_tokens


def compute_reconstruction_loss(original_windows, reconstructed_windows, reduction='mean'):
    """
    Compute reconstruction loss between original and reconstructed windows.
    
    Args:
        original_windows: Tensor of shape [batch, num_windows, input_dim]
        reconstructed_windows: Tensor of shape [batch, num_windows, input_dim]
        reduction: Reduction method ('mean', 'sum', or 'none')
    
    Returns:
        Reconstruction loss
    """
    return F.mse_loss(reconstructed_windows, original_windows, reduction=reduction)


def compute_compressor_loss(
    original_sequence,
    compressor,
    alpha=0.1,
    beta=0.01,
    reduction='mean'
):
    """
    Compute combined loss for compressor training.
    
    Args:
        original_sequence: Tensor of shape [batch, time_windows, channels, height, width]
                          with typical dimensions [B, T, 256, 8, 4]
        compressor: HierarchicalCompressorWithReconstruction instance
        alpha: Weight for diversity loss term
        beta: Weight for regularization loss term
        reduction: Reduction method for losses ('mean', 'sum', 'none')
    
    Returns:
        Tuple of (total_loss, recon_loss, diversity_loss):
            - total_loss: Combined weighted loss
            - recon_loss: Reconstruction loss (MSE) on channel dimension only
            - diversity_loss: Cosine similarity loss between output tokens
    
    Notes:
        The reconstruction loss only compares the channel dimension (C=256),
        not the full flattened spatial representation (C*H*W = 8192).
        This is extracted using original_sequence[:, :num_windows, :, 0, 0]
    """
    # Get exact dimensions
    batch_size, time_windows, channels, height, width = original_sequence.shape
    
    # Flatten input to the shape expected by the compressor
    flattened_input = original_sequence.reshape(batch_size, time_windows, channels * height * width)
    
    # Forward pass with reconstruction
    compressed, reconstructed, num_windows = compressor(
        flattened_input, with_reconstruction=True
    )
    
    # Extract channel dimension for reconstruction comparison
    # The model only reconstructs the channels (256) not the full C*H*W (8192)
    # Take the first element of each spatial position
    original_channels = original_sequence[:, :num_windows, :, 0, 0]
    
    # Compute reconstruction loss comparing just the channel dimension
    recon_loss = F.mse_loss(reconstructed, original_channels, reduction=reduction)
    
    # Compute diversity loss
    tokens = compressed.view(-1, compressed.size(-1))
    norm_tokens = F.normalize(tokens, p=2, dim=1)
    cosine_sim = torch.matmul(norm_tokens, norm_tokens.transpose(0, 1))
    mask = torch.eye(cosine_sim.size(0), device=cosine_sim.device)
    diversity_loss = (cosine_sim * (1.0 - mask)).mean()
    
    # Compute regularization loss
    reg_loss = beta * tokens.pow(2).mean()
    
    # Combine losses
    total_loss = recon_loss + alpha * diversity_loss + reg_loss
    
    return total_loss, recon_loss, diversity_loss