"""
Adapter module implementations for various temporal architectures
Each adapter transforms input features to output embeddings without knowledge of the surrounding system
"""

import torch
import torch.nn as nn


class LambdaLayer(nn.Module):
    """Simple Lambda layer for functional transformations"""
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
        
    def forward(self, x):
        return self.lambd(x)


class LinearAdapter(nn.Module):
    """Linear projection adapter
    
    Simple linear transformation from input dimensions to output dimensions
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)
    
    
class LSTMAdapter(nn.Module):
    """LSTM-based adapter for temporal sequence processing
    
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


class ConvolutionalAdapter(nn.Module):
    """Temporal convolutional adapter
    
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
        self.adapter = nn.Sequential(
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
        # Device handling is automatic since all operations in adapter
        # inherit from input
        return self.adapter(x)


class SelfAttentionAdapter(nn.Module):
    """Multi-block self-attention adapter with configurable attention pattern
    
    Processes sequences through multiple transformer blocks and projects to output dimension
    """
    
    def __init__(
        self, 
        input_dim, 
        output_dim,
        attention_mode,
        window_size,
        num_heads,
        num_layers,
        dropout,
    ):
        """
        Args:
            input_dim: Dimension of input features per timestep
            output_dim: Dimension of output features per timestep
            attention_mode: Type of attention pattern ('global', 'causal', 'local')
            window_size: Size of attention window for local attention
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create multiple transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=input_dim,
                nhead=num_heads,
                dropout=dropout,
                attention_mode=attention_mode,
                window_size=window_size
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
    
    def get_diagnostic_data(self, x):
        """
        Return diagnostic data about the adapter's operation
        
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
        
        # Forward through blocks, collecting diagnostics
        for i, block in enumerate(self.blocks):
            # Store input to this block
            layer_input = x.detach()
            
            # If this is a self-attention block with accessible attention weights
            if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'get_attention_weights'):
                # Forward through attention to get weights
                x = block(x)
                attn_weights = block.self_attn.get_attention_weights()
                diagnostic_data['attention_weights'].append(attn_weights)
            else:
                # Regular forward
                x = block(x)
            
            # Store output of this block
            diagnostic_data['layer_outputs'].append(x.detach())
        
        # Final projection
        x = self.projection(x)
        diagnostic_data['final_output'] = x.detach()
        
        return x, diagnostic_data


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture and configurable feedforward dimension."""
    
    def __init__(self,
                 embed_dim,
                 nhead,
                 dropout,
                 dim_feedforward=None,  # Allow configurable bottleneck dimension
                 attention_mode="global",
                 window_size=None):
        super().__init__()
        
        # Set bottleneck dimension (default to embed_dim // 4)
        dim_feedforward = dim_feedforward or embed_dim // 4
        
        # Pre-norm attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Pre-norm feedforward
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),  # Bottleneck
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),  # Project back
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.attention_mode = attention_mode
        self.window_size = window_size
        
    def forward(self, x):
        # Get attention mask
        seq_len = x.size(1)
        attn_mask = self._create_attention_mask(
            seq_len, self.attention_mode, self.window_size, device=x.device
        )
        
        # Pre-norm + Self-attention + Residual
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(
            query=x_norm, key=x_norm, value=x_norm,
            attn_mask=attn_mask
        )
        x = residual + self.dropout(attn_output)
        
        # Pre-norm + Feedforward + Residual
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.dropout(self.feedforward(x_norm))
        
        return x
    
    def _create_attention_mask(self, seq_len, attention_mode, window_size=None, device=None):
        """Create appropriate attention mask based on specified mode."""
        if attention_mode == "global":
            return None
            
        elif attention_mode == "causal":
            bool_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), 
                diagonal=1
            ).bool()
            float_mask = torch.zeros_like(bool_mask, dtype=torch.float)
            float_mask.masked_fill_(bool_mask, float('-inf'))
            return float_mask
            
        elif attention_mode == "local" and window_size is not None:
            bool_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                bool_mask[i, start:end] = False
            float_mask = torch.zeros_like(bool_mask, dtype=torch.float)
            float_mask.masked_fill_(bool_mask, float('-inf'))
            return float_mask
            
        return None


class RNNAdapter(nn.Module):
    """
    RNN adapter for processing feature sequences
    
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
        Forward pass through the RNN adapter
        
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