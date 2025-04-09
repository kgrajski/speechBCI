"""
Unified adapter for transformer-based models (T5, BART)
"""

import torch
from torch import nn
from mmllm.model_adapters.adapter_modules import (
    LinearAdapter, LSTMAdapter, RNNAdapter,
    ConvolutionalAdapter as ConvAdapter,
    SelfAttentionAdapter as AttentionAdapter
)

class TransformerAdapter(nn.Module):
    """
    Universal adapter for transformer models with similar API (T5, BART)
    """

    def __init__(
        self, 
        input_dim,  # Changed from embed_dim for consistency
        output_dim,  # Changed from hidden_size to match adapter_modules.py
        attention_mode=None,
        window_size=None,
        total_input_dim=None,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    ):
        """
        Initialize a standalone adapter that transforms input features to output dimensions.

        Args:
            input_dim: Dimension of features at each timestep
            output_dim: Target output dimension for transformed features
            attention_mode: Attention pattern ('global', 'causal', 'local')
            window_size: Size of attention window for local attention
            total_input_dim: Total flattened dimension for linear adapter
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        # Store dimensions
        self.input_dim = input_dim  # Updated variable name
        self.output_dim = output_dim  # Updated variable name
        
        # Determine which adapter implementation to use (defaulting to attention)
        adapter_type = "attention" if attention_mode is not None else "linear"
        self.adapter_type = adapter_type.lower()
        
        # Configure the adapter based on type
        if self.adapter_type == "linear":
            # For linear adapters, input_dim represents the total flattened dimension
            linear_input_dim = total_input_dim if total_input_dim else input_dim
            self.adapter = LinearAdapter(linear_input_dim, output_dim)
        elif self.adapter_type == "lstm":
            self.adapter = LSTMAdapter(input_dim, output_dim)
        elif self.adapter_type == "conv":
            self.adapter = ConvAdapter(input_dim, output_dim)
        elif self.adapter_type == "attention":
            self.adapter = AttentionAdapter(
                input_dim=input_dim,  # Updated parameter name  
                output_dim=output_dim, 
                attention_mode=attention_mode, 
                window_size=window_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif self.adapter_type == "rnn":
            self.adapter = RNNAdapter(input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")

    def forward(self, inputs):
        """
        Forward pass through the adapter only.
        
        Args:
            inputs: Input features with shape:
                - For sequence models: [batch_size, seq_len, input_dim]
                - For linear adapter: [batch_size, total_input_dim]

        Returns:
            Tensor with shape [batch_size, seq_len, output_dim] 
        """
        # Transform input features to output dimensions
        return self.adapter(inputs)
        
    def print_trainable_parameters(self):
        """Print the number of trainable parameters in the adapter."""
        total_params = 0
        trainable_params = 0
        
        for param in self.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        print(f"Adapter Total Parameters: {total_params:,}")
        print(f"Adapter Trainable Parameters: {trainable_params:,}")
        print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%")