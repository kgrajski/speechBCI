"""
Hierarchical Attention Compressor for neural BCI data.

This module implements a hierarchical compression approach for VQ-VAE embeddings,
processing first at the spatial level (8×4 grid) and then compressing temporally
to create a fixed-length sequence suitable for language model processing.

References:
----------
1. Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). 
   "Hierarchical Attention Networks for Document Classification." 
   Proceedings of NAACL-HLT 2016.
   - Introduces the concept of hierarchical attention for processing structured data

2. Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A., & Carreira, J. (2021). 
   "Perceiver: General Perception with Iterative Attention." 
   International Conference on Machine Learning, 4651-4664.
   - Presents a fixed-latent architecture for processing arbitrary input lengths

3. Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). 
   "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks." 
   International Conference on Machine Learning, 3744-3753.
   - Introduces inducing points (similar to our target queries) for set compression

4. Rae, J. W., Potapenko, A., Jayakumar, S. M., & Lillicrap, T. P. (2019). 
   "Compressive Transformers for Long-Range Sequence Modelling." 
   International Conference on Learning Representations.
   - Provides techniques for compressing long sequences with transformers

5. Wang, X., Han, C., Chen, Y., & Wang, Z. (2022). 
   "BrainBERT: A Transformer-based Framework for EEG-based Brain-Computer Interfaces." 
   IEEE Transactions on Neural Systems and Rehabilitation Engineering, 30, 1704-1713.
   - Applies transformer architectures to neural data (specifically EEG)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalAttentionCompressor(nn.Module):
    """
    Hierarchical Attention Compressor for neural data.
    
    This compressor processes VQ-VAE embeddings in two stages:
    1. Spatial Stage: Processes each time window's spatial tokens (8×4 grid) via attention
    2. Temporal Stage: Compresses the resulting sequence to a fixed length using
       transformer encoders and learned queries
    
    The output maintains the original embedding dimension while reducing sequence length.
    """
    
    def __init__(
        self,
        input_dim=64,             # Dimension of VQ-VAE embeddings
        hidden_dim=256,           # Internal representation dimension
        output_dim=None,          # Output dimension (defaults to input_dim)
        spatial_shape=(8, 4),     # Shape of spatial grid (H×W)
        output_tokens=512,        # Number of output tokens
        num_heads_spatial=4,      # Number of attention heads for spatial processing
        num_heads_temporal=8,     # Number of attention heads for temporal processing
        num_layers_temporal=2,    # Number of transformer layers for temporal processing
        max_input_windows=250,    # Maximum number of time windows to process
        dropout=0.1               # Dropout rate
    ):
        """
        Initialize the hierarchical compressor.
        
        Args:
            input_dim: Dimension of input VQ-VAE embeddings
            hidden_dim: Dimension of hidden representations
            output_dim: Dimension of output embeddings (defaults to input_dim)
            spatial_shape: Tuple of (height, width) for spatial arrangement
            output_tokens: Number of output tokens to produce
            num_heads_spatial: Number of attention heads for spatial processing
            num_heads_temporal: Number of attention heads for temporal processing
            num_layers_temporal: Number of transformer layers for temporal processing
            max_input_windows: Maximum number of time windows to process
            dropout: Dropout rate for attention layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.spatial_shape = spatial_shape
        self.spatial_tokens = spatial_shape[0] * spatial_shape[1]
        self.output_tokens = output_tokens
        self.max_input_windows = max_input_windows
        self.dropout = dropout
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Spatial position encoding
        self.register_spatial_encoding()
        
        # Learnable summary token for each time window
        self.spatial_summary = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Spatial attention for processing tokens within each time window
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads_spatial,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for spatial processing
        self.spatial_norm1 = nn.LayerNorm(hidden_dim)
        self.spatial_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network for spatial processing
        self.spatial_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Temporal transformer for processing across time windows
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads_temporal,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_encoder_layer,
            num_layers=num_layers_temporal
        )
        
        # Learnable queries for output token selection
        self.target_queries = nn.Parameter(torch.randn(output_tokens, hidden_dim))
        self.query_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)
    
    def register_spatial_encoding(self):
        """Create 2D spatial position encodings for the grid layout."""
        h, w = self.spatial_shape
        
        # Create position encodings for height and width dimensions
        pos_h = torch.arange(h).float().unsqueeze(1)
        pos_w = torch.arange(w).float().unsqueeze(1)
        
        # Scale positions to [0, 1]
        pos_h = pos_h / max(h - 1, 1)
        pos_w = pos_w / max(w - 1, 1)
        
        # Create position encodings with sin/cos
        dim = self.hidden_dim // 4  # Split hidden dim for h/w and sin/cos
        
        # FIXED: Remove step=2 to generate the correct number of frequencies
        div_term = torch.exp(torch.arange(0, dim).float() * -(math.log(10000.0) / dim))
        
        # Calculate encodings for height
        pos_h_enc = torch.zeros(h, self.hidden_dim // 2)
        pos_h_enc[:, 0::2] = torch.sin(pos_h * div_term)
        pos_h_enc[:, 1::2] = torch.cos(pos_h * div_term)
        
        # Calculate encodings for width
        pos_w_enc = torch.zeros(w, self.hidden_dim // 2)
        pos_w_enc[:, 0::2] = torch.sin(pos_w * div_term)
        pos_w_enc[:, 1::2] = torch.cos(pos_w * div_term)
        
        # Combine into grid
        pos_enc = torch.zeros(h * w, self.hidden_dim)
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                pos_enc[idx, :self.hidden_dim // 2] = pos_h_enc[i]
                pos_enc[idx, self.hidden_dim // 2:] = pos_w_enc[j]
        
        # Register as buffer (non-parameter tensor)
        self.register_buffer('spatial_encoding', pos_enc.unsqueeze(0))
    
    def process_spatial_window(self, window_tokens):
        """
        Process spatial tokens from a single time window.
        
        Args:
            window_tokens: Tensor of shape [batch_size, spatial_tokens, hidden_dim]
        
        Returns:
            Tensor of shape [batch_size, hidden_dim] representing the window summary
        """
        batch_size = window_tokens.shape[0]
        
        # Add spatial position encoding
        window_tokens = window_tokens + self.spatial_encoding
        
        # Prepare summary token
        summary = self.spatial_summary.expand(batch_size, -1, -1)
        
        # Combine summary with spatial tokens
        combined = torch.cat([summary, window_tokens], dim=1)
        
        # Apply self-attention (summary attends to all spatial tokens)
        summary_idx = torch.zeros(batch_size, 1, combined.size(1), device=combined.device)
        summary_idx[:, :, 0] = 1  # Focus on first token (summary)
        
        # First attention block (summary token as query)
        # The summary token attends to all spatial tokens in the window
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
            x: Input tensor of shape [batch, time_windows, channels, height, width]
            with_reconstruction: If True, also return reconstructed time windows
        
        Returns:
            If with_reconstruction is False:
                Tensor of shape [batch, output_tokens, output_dim]
            If with_reconstruction is True:
                Tuple of (compressed_tokens, reconstructed_windows, num_windows)
        """
        # Check and reshape input if it's 5D
        if len(x.shape) == 5:
            batch_size, time_windows, channels, height, width = x.shape
            x = x.reshape(batch_size, time_windows, channels * height * width)
        
        batch_size = x.size(0)
        
        # Prepare input sequence
        x_prepared, num_windows = self.prepare_input_sequence(x)
        
        # Rest of processing remains the same...
        if num_windows == 0:
            empty_tokens = torch.zeros(
                batch_size, self.output_tokens, self.output_dim, 
                device=x.device
            )
            if with_reconstruction:
                empty_recon = torch.zeros(
                    batch_size, 0, self.input_dim,
                    device=x.device
                )
                return empty_tokens, empty_recon, 0
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
        output_tokens = self.output_norm(output_tokens)
        
        if with_reconstruction:
            # Project compressed tokens back to hidden dimension
            compressed_hidden = self.input_projection(output_tokens)
            
            # Reconstruct time window representations
            reconstructed_windows = self.reconstruct(compressed_hidden, num_windows)
            
            return output_tokens, reconstructed_windows, num_windows
        
        return output_tokens


class HierarchicalCompressorWithReconstruction(HierarchicalAttentionCompressor):
    """
    Extended version of the hierarchical compressor with reconstruction capability.
    
    This version adds the ability to reconstruct the original time window representations
    from the compressed tokens, which is useful for pretraining with an auxiliary loss.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as parent class."""
        super().__init__(*args, **kwargs)
        
        # Add reconstruction-specific components
        self.reconstructor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True
            ),
            num_layers=2
        )
        
        # Position embeddings for reconstruction targets
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.max_input_windows, self.hidden_dim)
        )
        
        # Output projection for reconstruction
        self.reconstruction_projection = nn.Linear(self.hidden_dim, self.input_dim)
        self.reconstruction_norm = nn.LayerNorm(self.input_dim)
    
    def reconstruct(self, compressed_tokens, num_windows):
        """
        Reconstruct original time window representations from compressed tokens.
        
        Args:
            compressed_tokens: Tensor of shape [batch, output_tokens, hidden_dim]
            num_windows: Number of time windows to reconstruct
        
        Returns:
            Tensor of shape [batch, num_windows, input_dim]
        """
        batch_size = compressed_tokens.size(0)
        
        # Get position embeddings for target sequence
        positions = self.position_embeddings[:, :num_windows].expand(batch_size, -1, -1)
        
        # Reconstruct using transformer decoder
        # positions are the query (what we want to reconstruct)
        # compressed_tokens are the memory (what we use for reconstruction)
        reconstructed = self.reconstructor(positions, compressed_tokens)
        
        # Project to input dimension
        reconstructed = self.reconstruction_projection(reconstructed)
        reconstructed = self.reconstruction_norm(reconstructed)
        
        return reconstructed


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
        compressor: HierarchicalCompressorWithReconstruction instance
        alpha: Weight for diversity loss
        beta: Weight for regularization loss
        reduction: Reduction method for losses
    
    Returns:
        Tuple of (total_loss, recon_loss, diversity_loss)
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