"""
Hierarchical Attention Compressor for neural time series data.

This module implements a hierarchical attention-based architecture for compressing
neural time series data into fixed-length representations suitable for language modeling.

References:
- Vaswani et al. "Attention Is All You Need" (2017)
- Yang et al. "Hierarchical Attention Networks for Document Classification" (2016)
- Child et al. "Generating Long Sequences with Sparse Transformers" (2019)
"""

"""
Understanding Token Diversity Loss in Our Model
What the Diversity Loss Measures
Our enhanced token diversity loss combines two metrics:

Coverage Loss (70% weight):

Measures what percentage of your token vocabulary is being used
Range: 0.0 (all tokens used) to 1.0 (only one token used)
Uniformity Loss (30% weight):

Measures how evenly distributed the token usage is
Based on entropy of token distribution
Range: 0.0 (perfectly uniform) to 1.0 (completely skewed)

Expected Behavior During Training
Unlike perplexity which should increase over time, our diversity loss should decrease as training progresses:

Training Stage	Expected Value	Interpretation
Initial	0.8-0.95	Model is using very few tokens (mode collapse)
Mid-training	0.4-0.7	Model is starting to utilize more tokens
Well-trained	0.1-0.3	Model is using a diverse set of tokens with good distribution

Diagnostic Values to Track
In TensorBoard, pay attention to these metrics:

token_coverage: Should increase from ~5-10% to 60-80%
normalized_entropy: Should increase from near 0 to 0.6-0.8
tokens_used: The raw count should increase steadily
Warning Signs
Stuck high (>0.8): Severe mode collapse - model is using very few tokens
Too low too quickly (<0.1): Might be sacrificing reconstruction quality
Fluctuating widely: Unstable training, possibly too high learning rate
Plateauing early: Might need to increase alpha (diversity weight)

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        
        # Add a projection for reconstruction
        self.recon_projection = nn.Linear(hidden_dim, input_dim)
    
    def initialize_codebook(self):
        """Initialize a codebook for vector quantization."""
        # Create a codebook of size [output_tokens, output_dim]
        codebook = torch.randn(self.output_tokens, self.output_dim) * 0.02
        self.register_parameter("codebook", nn.Parameter(codebook))
        
        # Register buffers for EMA tracking (not parameters)
        # Using register_buffer properly keeps them with the model across device moves
        self.register_buffer("ema_count", torch.zeros(self.output_tokens))
        self.register_buffer("ema_weight", torch.zeros(self.output_tokens, self.output_dim))
        self.register_buffer("ema_initialized", torch.tensor(0, dtype=torch.bool))

    def quantize_tokens(self, token_embeddings):
        """
        Quantize continuous token embeddings to nearest vectors in codebook.
        """
        # Initialize codebook if it doesn't exist
        if not hasattr(self, "codebook"):
            self.initialize_codebook()
        
        # Get device from input
        device = token_embeddings.device
        
        # Don't try to modify the parameter directly - instead create a local copy on the right device
        codebook = self.codebook.to(device)
        
        # Reshape inputs for easier processing
        batch_size, num_tokens, dim = token_embeddings.shape
        flat_inputs = token_embeddings.reshape(-1, dim)  # [batch*tokens, dim]
        
        # Calculate distances to all codebook vectors
        # Normalize embeddings for cosine distance
        norm_inputs = F.normalize(flat_inputs, p=2, dim=1)
        norm_codebook = F.normalize(codebook, p=2, dim=1)
        
        # Compute cosine similarity (higher is closer)
        similarity = torch.matmul(norm_inputs, norm_codebook.transpose(0, 1))
        
        # Get indices of nearest vectors
        token_indices = torch.argmax(similarity, dim=1)
        token_indices = token_indices.view(batch_size, num_tokens)
        
        # Get the corresponding vectors from the codebook
        flat_indices = token_indices.reshape(-1)
        quantized = codebook[flat_indices]
        quantized = quantized.reshape(batch_size, num_tokens, dim)
        
        # Comment out or disable the EMA update call 
        # self.update_codebook_with_ema(token_indices.view(-1), flat_inputs)
        
        return token_indices, quantized
    
    def update_codebook_with_ema(self, encodings, flat_inputs, decay=0.99):
        """Temporarily disabled EMA update to resolve device issues"""
        # Skip the update entirely during initial debugging
        return
    
    def compute_token_diversity_loss(self, token_indices):
        """Compute enhanced token diversity loss."""
        batch_size = token_indices.shape[0]
        device = token_indices.device
        
        # Create histogram of token usage
        token_usage = torch.zeros(batch_size, self.output_tokens, device=device)
        for b in range(batch_size):
            unique_tokens, counts = torch.unique(token_indices[b], return_counts=True)
            token_usage[b, unique_tokens] = counts.float()
        
        # Calculate per-batch token coverage (what % of tokens are used at all)
        tokens_used = torch.sum(token_usage > 0, dim=1).float()
        token_coverage = tokens_used / self.output_tokens
        coverage_loss = 1.0 - token_coverage.mean()
        
        # Calculate distribution uniformity using entropy
        probs = token_usage / (torch.sum(token_usage, dim=1, keepdim=True) + 1e-10)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        max_entropy = torch.log(torch.tensor(self.output_tokens, dtype=torch.float, device=device))
        uniformity_loss = 1.0 - (entropy / max_entropy).mean()
        
        # Combined loss with stronger weight on coverage
        diversity_loss = 0.7 * coverage_loss + 0.3 * uniformity_loss
        
        # Return usage statistics for monitoring
        usage_stats = {
            'tokens_used': tokens_used.mean().item(),
            'token_coverage': token_coverage.mean().item(),
            'entropy': entropy.mean().item(),
            'normalized_entropy': (entropy / max_entropy).mean().item()
        }
        
        return diversity_loss, usage_stats
    
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
    
    def apply_token_dropout(self, token_indices, quantized_tokens, p=0.1):
        """Temporarily disabled token dropout"""
        # Just return the inputs unchanged to bypass this feature
        return token_indices, quantized_tokens
    
    def forward(self, x, with_reconstruction=True):
        """
        Forward pass through the hierarchical compressor.
        
        Args:
            x: Input tensor of shape [batch_size, time_windows, input_dim*spatial_h*spatial_w]
            with_reconstruction: Whether to generate reconstruction output
        
        Returns:
            Dictionary containing:
            - compressed: Quantized tokens [batch_size, output_tokens, output_dim]
            - continuous: Continuous tokens [batch_size, output_tokens, output_dim]
            - reconstructed: Reconstructed input (if with_reconstruction=True)
            - recon_loss: Reconstruction loss (if with_reconstruction=True)
            - diversity_loss: Diversity loss based on token similarity
            - enhanced_div_loss: Enhanced diversity loss based on token usage
            - reg_loss: Regularization loss
            - token_indices: Indices of selected tokens
            - token_stats: Statistics about token usage
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
                return {
                    'compressed': empty_tokens,
                    'continuous': empty_tokens,
                    'reconstructed': empty_windows,
                    'recon_loss': None,
                    'diversity_loss': None,
                    'enhanced_div_loss': None,
                    'reg_loss': None,
                    'token_indices': None,
                    'token_stats': None
                }
            return {'compressed': empty_tokens, 'continuous': empty_tokens}
        
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
        intermediate_output = torch.einsum('btn,bnd->btd', attn_weights, temporal_features)
        
        # Project to output dimension
        compressed = self.output_projection(intermediate_output)
        
        # Quantize the tokens - get discrete tokens and indices
        token_indices, quantized_tokens = self.quantize_tokens(compressed)
        
        # Apply token dropout during training
        if self.training:
            token_indices, quantized_tokens = self.apply_token_dropout(token_indices, quantized_tokens, p=0.15)
        
        # Compute enhanced token diversity loss
        enhanced_div_loss, token_stats = self.compute_token_diversity_loss(token_indices)
        
        # Calculate standard diversity loss for comparison
        tokens = compressed.view(-1, compressed.size(-1))
        norm_tokens = F.normalize(tokens, p=2, dim=1)
        cosine_sim = torch.matmul(norm_tokens, norm_tokens.transpose(0, 1))
        mask = torch.eye(cosine_sim.size(0), device=cosine_sim.device)
        diversity_loss = (cosine_sim * (1.0 - mask)).mean()
        
        # Calculate regularization loss
        reg_loss = tokens.pow(2).mean()
        
        # Handle reconstruction if needed
        reconstructed = None
        recon_loss = None
        if with_reconstruction:
            # First compute mean representation across time
            recon_features = temporal_features.mean(dim=1, keepdim=True)  # [batch, 1, hidden_dim]
            
            # Project back to input dimension
            recon_features = self.recon_projection(recon_features)  # [batch, 1, input_dim]
            
            # Expand to match number of windows
            reconstructed = recon_features.expand(-1, num_windows, -1)  # [batch, num_windows, input_dim]
            
            # Now dimensions will match for the loss
            recon_loss = F.mse_loss(reconstructed, x[:, :num_windows, :self.input_dim], reduction='mean')
        
        # Return all outputs
        return {
            'compressed': quantized_tokens,  # Use quantized tokens as output
            'continuous': compressed,        # Also return continuous version
            'reconstructed': reconstructed,
            'recon_loss': recon_loss,
            'diversity_loss': diversity_loss,
            'enhanced_div_loss': enhanced_div_loss,
            'reg_loss': reg_loss,
            'token_indices': token_indices,
            'token_stats': token_stats
        }


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
            Dictionary containing compressed tokens and other outputs
        """
        # Get outputs from parent class (now returns a dictionary)
        outputs = super().forward(x, with_reconstruction=True)
        
        # Extract what we need
        compressed_tokens = outputs['compressed']
        
        if with_reconstruction:
            # We already have reconstructed outputs from the parent class
            return outputs
        
        # For backward compatibility with code expecting just compressed tokens
        return outputs


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