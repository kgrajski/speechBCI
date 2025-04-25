"""
Encoder module implementations for various temporal architectures
Each encoder transforms input features to output embeddings without knowledge of the surrounding system
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture and configurable feedforward dimension."""
    
    def __init__(self,
                 input_dim,
                 nhead,
                 dropout,
                 dim_feedforward=None,  # Allow configurable bottleneck dimension
                 attention_pattern="global",  # Renamed from attention_mode to attention_pattern
                 window_size=None,  # Window size for local attention pattern
                 enable_diagnostics=False):  # Add diagnostics flag
        super().__init__()
        self.nhead = nhead
        self.attention_pattern = attention_pattern  # Renamed from attention_mode
        self.window_size = window_size
        self.enable_diagnostics = enable_diagnostics
        
        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Self-attention with configurable attention pattern
        self.self_attn = nn.MultiheadAttention(
            input_dim, nhead, dropout=dropout, batch_first=True
        )
        
        # Feedforward network with configurable bottleneck
        if dim_feedforward is None:
            dim_feedforward = input_dim // 4  # Default bottleneck factor of 4
            
        self.linear1 = nn.Linear(input_dim, dim_feedforward)  # Project down
        self.linear2 = nn.Linear(dim_feedforward, input_dim)  # Project back up
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def _create_attention_pattern_mask(self, seq_len, attention_pattern, window_size=None, device=None):
        """Create appropriate attention pattern mask based on specified pattern."""
        if attention_pattern == "global":
            return None
            
        elif attention_pattern == "causal":
            bool_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            return bool_mask.masked_fill(bool_mask == 1, float('-inf'))
            
        elif attention_pattern == "local":
            if window_size is None:
                raise ValueError("window_size must be specified for local attention pattern")
            # Create local attention pattern mask
            mask = torch.ones(seq_len, seq_len, device=device)
            for i in range(seq_len):
                start = max(0, i - window_size)
                end = min(seq_len, i + window_size + 1)
                mask[i, start:end] = 0
            return mask.masked_fill(mask == 1, float('-inf'))
            
        else:
            raise ValueError(f"Unsupported attention pattern: {attention_pattern}")
            
    def forward(self, x, padding_masks=None):
        """Forward pass with optional padding mask."""
        # Get sequence length
        seq_len = x.size(1)
        
        # Create attention pattern mask
        attn_pattern_mask = self._create_attention_pattern_mask(
            seq_len, self.attention_pattern, self.window_size, device=x.device
        )
        
        # Handle padding mask
        if padding_masks is not None:
            # Ensure padding_masks is bool type
            padding_masks = padding_masks.bool()
            # For MultiheadAttention, True means ignore
            key_padding_masks = ~padding_masks
        else:
            key_padding_maskss = None
            
        # Pre-norm + Self-attention + Residual
        residual = x
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.self_attn(
            query=x_norm, 
            key=x_norm, 
            value=x_norm,
            attn_mask=attn_pattern_mask,  # Use attention pattern mask directly
            key_padding_mask=key_padding_masks  # Use padding mask directly
        )
        
        if self.enable_diagnostics:
            # Log attention weight statistics
            print(f"Attention weights shape: {attn_weights.shape}")
            print(f"Mean attention weight: {attn_weights.mean().item():.4f}")
            print(f"Std attention weight: {attn_weights.std().item():.4f}")
            
            # Log attention to padding positions
            if padding_masks is not None:
                padding_attention = attn_weights[~padding_masks].mean().item()
                print(f"Mean attention to padding: {padding_attention:.4f}")
        
        x = residual + self.dropout(attn_output)
        
        # Feedforward + Residual
        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout(x)
        
        return x
    
    
class SelfAttentionEncoder(nn.Module):
    """Multi-block self-attention encoder with configurable attention pattern
    
    Processes sequences through multiple transformer blocks and projects to output dimension
    """
    
    def __init__(
        self, 
        input_dim, 
        output_dim,
        attention_pattern,
        window_size,
        num_heads,
        num_layers,
        dropout,
    ):
        """
        Args:
            input_dim: Dimension of input features per timestep
            output_dim: Dimension of output features per timestep
            attention_pattern: Type of attention pattern ('global', 'causal', 'local')
            window_size: Size of attention window for local attention pattern
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # First block handles padding mask
        self.first_block = TransformerBlock(
            input_dim=input_dim,
            nhead=num_heads,
            dropout=dropout,
            attention_pattern=attention_pattern,
            window_size=window_size
        )
        
        # Subsequent blocks don't need padding mask handling
        self.blocks = nn.ModuleList([
            TransformerBlock(
                input_dim=input_dim,
                nhead=num_heads,
                dropout=dropout,
                attention_pattern=attention_pattern,
                window_size=window_size
            ) for _ in range(num_layers - 1)  # One less since we have first_block
        ])
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            #nn.LayerNorm(output_dim)
        )
        
    def forward(self, x, padding_masks=None):
        # First block handles padding mask
        x = self.first_block(x, padding_masks)
        
        # Subsequent blocks don't use padding mask
        for block in self.blocks:
            x = block(x)
            
        # Final projection
        return self.projection(x)
    
    def get_diagnostic_data(self, x):
        """
        Return diagnostic data about the encoder's operation
        
        Args:
            x: Input tensor
            
        Returns:
            tuple: (output tensor, diagnostic data dictionary)
        """
        diagnostic_data = {
            'input_shape': x.shape,
            'attention_weights': [],
            'layer_outputs': []
        }
        
        # Forward through first block
        layer_input = x.detach()
        x = self.first_block(x)
        if hasattr(self.first_block.self_attn, 'get_attention_weights'):
            attn_weights = self.first_block.self_attn.get_attention_weights()
            diagnostic_data['attention_weights'].append(attn_weights)
        diagnostic_data['layer_outputs'].append(x.detach())
        
        # Forward through remaining blocks
        for block in self.blocks:
            layer_input = x.detach()
            x = block(x)
            if hasattr(block.self_attn, 'get_attention_weights'):
                attn_weights = block.self_attn.get_attention_weights()
                diagnostic_data['attention_weights'].append(attn_weights)
            diagnostic_data['layer_outputs'].append(x.detach())
        
        # Final projection
        x = self.projection(x)
        diagnostic_data['final_output'] = x.detach()
        
        return x, diagnostic_data


class RNNEncoder(nn.Module):
    """
    RNN encoder for processing feature sequences
    
    Transforms input sequences using simple RNN layers and projects to output dimension
    """
    def __init__(self, input_dim, output_dim, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim: Dimension of input features per timestep
            output_dim: Dimension of output features per timestep
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=output_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.projection = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the RNN encoder
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        output, _ = self.rnn(x)
        output = self.projection(output)
        output = self.norm(output)
        output = self.dropout(output)
        return output
    
class LambdaLayer(nn.Module):
    """Simple Lambda layer for functional transformations"""
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
        
    def forward(self, x):
        return self.lambd(x)
    
class LinearBlock(nn.Module):
    """Linkear block with pre-norm architecture and configurable feedforward dimension."""
    
    def __init__(self,
                 input_dim,
                 dropout,
                 dim_feedforward=None,
                 ):
        super().__init__()
        
        # Set bottleneck dimension (default to input_dim // 4)
        dim_feedforward = dim_feedforward or input_dim // 4
        
        # Pre-norm attention
        self.norm1 = nn.LayerNorm(input_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),  # Bottleneck
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),  # Project back
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):

        return self.dropout(self.feedforward(self.norm1(x)))


class LinearEncoder(nn.Module):
    """Linear projection encoder
    
    Simple linear transformation from input dimensions to output dimensions
    """
    
    def __init__(
        self, 
        input_dim, 
        output_dim,
        num_layers,
        dropout,
    ):
        """
        Args:
            input_dim: Dimension of input features per timestep
            output_dim: Dimension of output features per timestep
            num_layers: Number of transformer blocks
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create multiple transformer blocks
        self.blocks = nn.ModuleList([
            LinearBlock(
                input_dim=input_dim,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        # Pass through each transformer block
        for block in self.blocks:
            x = block(x)
            
        # Final projection
        return self.projection(x)
    
class LSTMEncoder(nn.Module):
    """LSTM-based encoder for temporal sequence processing
    
    Processes temporal sequences through bidirectional LSTM and projects to output dimension
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim: Dimension of input features per timestep
            output_dim: Dimension of output features per timestep
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            batch_first=True,
            bidirectional=True
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim*2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        # Device handling is automatic since LSTM inherits from input
        outputs, _ = self.lstm(x)
        return self.projection(outputs)


class ConvolutionalEncoder(nn.Module):
    """Temporal convolutional encoder
    
    Processes sequence with 1D convolution along time dimension
    """
    
    def __init__(self, input_dim, output_dim, kernel_size=3):
        """
        Args:
            input_dim: Dimension of input features per timestep
            output_dim: Dimension of output features per timestep
            kernel_size: Size of convolution kernel
        """
        super().__init__()
        self.encoder = nn.Sequential(
            # Reshape to [B, C, T]
            LambdaLayer(lambda x: x.transpose(1, 2)),
            # 1D convolution along time dimension
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=kernel_size,
                padding=kernel_size//2
            ),
            # Reshape back to [B, T, C]
            LambdaLayer(lambda x: x.transpose(1, 2)),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        # Device handling is automatic since all operations in encoder
        # inherit from input
        return self.encoder(x)