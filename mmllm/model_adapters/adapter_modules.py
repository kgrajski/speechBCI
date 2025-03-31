"""
Adapter module implementations for various temporal architectures
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

class LSTMAdapter(nn.Module):
    """LSTM-based adapter for temporal sequence processing"""
    
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
            bidirectional=True
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim*2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        # Device handling is automatic since LSTM inherits from input
        outputs, _ = self.lstm(x)
        return self.projection(outputs)

class ConvolutionalAdapter(nn.Module):
    """Temporal convolutional adapter"""
    
    def __init__(self, embedding_dim, output_dim, kernel_size=3):
        super().__init__()
        self.adapter = nn.Sequential(
            # Reshape to [B, C, T]
            LambdaLayer(lambda x: x.transpose(1, 2)),
            # 1D convolution along time dimension
            nn.Conv1d(
                in_channels=embedding_dim,
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
    """Self-attention based temporal adapter with configurable attention pattern"""
    
    def __init__(self, embedding_dim, output_dim, num_heads=4, attention_mode="global", window_size=None):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            output_dim: Dimension of output features
            num_heads: Number of attention heads
            attention_mode: Type of attention pattern ('global', 'causal', 'local')
            window_size: Size of attention window for local attention (None = full sequence)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.attention_mode = attention_mode
        self.window_size = window_size
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def _create_attention_mask(self, seq_len, attention_mode, window_size=None, device=None):
        """Create appropriate attention mask based on specified mode"""
        if attention_mode == "global":
            # No mask for global attention
            return None
            
        elif attention_mode == "causal":
            # Create causal mask
            bool_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), 
                diagonal=1
            ).bool()
            # Convert to float mask with -inf for masked positions
            float_mask = torch.zeros_like(bool_mask, dtype=torch.float)
            float_mask.masked_fill_(bool_mask, float('-inf'))
            return float_mask
            
        elif attention_mode == "local" and window_size is not None:
            # Create local window mask
            bool_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                bool_mask[i, start:end] = False
            # Convert to float mask with -inf for masked positions
            float_mask = torch.zeros_like(bool_mask, dtype=torch.float)
            float_mask.masked_fill_(bool_mask, float('-inf'))
            return float_mask
            
        return None
        
    def forward(self, x):
        # x shape: [batch, seq_len, embedding_dim]
        seq_len = x.size(1)
        
        # Create attention mask based on mode, using input tensor's device
        attn_mask = self._create_attention_mask(
            seq_len, 
            self.attention_mode, 
            self.window_size,
            device=x.device  # Pass device from input tensor
        )
        
        # Apply self-attention with appropriate mask
        attn_out, _ = self.self_attn(
            query=x, 
            key=x, 
            value=x, 
            attn_mask=attn_mask,
            key_padding_mask=None
        )
        
        return self.projection(attn_out)

class RNNAdapter(nn.Module):
    """
    Standard RNN adapter for processing embedding sequences.
    
    This adapter uses a simple RNN layer followed by a projection to transform
    input embeddings before passing them to the language model.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the RNN adapter.
        """
        output, _ = self.rnn(x)
        output = self.projection(output)
        output = self.norm(output)
        output = self.dropout(output)
        return output
