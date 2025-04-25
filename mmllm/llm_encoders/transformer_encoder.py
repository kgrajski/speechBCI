"""
Unified Encoder for transformer-based models (T5, BART)
"""

import torch
from torch import nn
from mmllm.llm_encoders.encoder_modules import (
    LinearEncoder, LSTMEncoder, RNNEncoder,
    ConvolutionalEncoder as ConvEncoder,
    SelfAttentionEncoder as AttentionEncoder
)

class TransformerEncoder(nn.Module):
    """
    Universal encoder for transformer models with similar API (T5, BART)
    """

    def __init__(
        self, 
        input_dim,  # Changed from embed_dim for consistency
        output_dim,  # Changed from hidden_size to match Encoder_modules.py
        encoder_type,
        attention_mode,
        window_size,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    ):
        """
        Initialize a standalone encoder that transforms input features to output dimensions.

        Args:
            input_dim: Dimension of features at each timestep
            output_dim: Target output dimension for transformed features
            attention_mode: Attention pattern ('global', 'causal', 'local')
            window_size: Size of attention window for local attention
            total_input_dim: Total flattened dimension for linear encoder
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        # Store dimensions
        self.input_dim = input_dim  # Updated variable name
        self.output_dim = output_dim  # Updated variable name
        
        # Determine which encoder implementation to use (defaulting to attention)
        self.encoder_type = encoder_type.lower()
        
        # Configure the encoder based on type
        if self.encoder_type == "linear":
            total_input_dim = None  # Placeholder for total input dimension
            # For linear encoders, input_dim represents the total flattened dimension
            linear_input_dim = total_input_dim if total_input_dim else input_dim
            self.encoder = LinearEncoder(linear_input_dim, output_dim, num_layers, dropout)
        elif self.encoder_type == "lstm":
            self.encoder = LSTMEncoder(input_dim, output_dim)
        elif self.encoder_type == "conv":
            self.encoder = ConvEncoder(input_dim, output_dim)
        elif self.encoder_type == "attention":
            self.encoder = AttentionEncoder(
                input_dim=input_dim,  # Updated parameter name  
                output_dim=output_dim, 
                attention_pattern=attention_mode,  # Renamed from attention_mode to attention_pattern
                window_size=window_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif self.encoder_type == "rnn":
            self.encoder = RNNEncoder(input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def forward(self, inputs, padding_masks=None, labels=None):
        """
        Forward pass through the encoder only.
        
        Args:
            inputs: Input features with shape:
                - For sequence models: [batch_size, seq_len, input_dim]
                - For linear encoder: [batch_size, total_input_dim]
            padding_masks: Optional mask indicating which positions are real data (1) vs padding (0)
            labels: Optional target labels for training

        Returns:
            Tensor with shape [batch_size, seq_len, output_dim] 
        """
        # Transform input features to output dimensions
        if self.encoder_type == "attention":
            return self.encoder(inputs, padding_masks=padding_masks)
        else:
            return self.encoder(inputs)
        
    def print_trainable_parameters(self):
        """Print the number of trainable parameters in the encoder."""
        total_params = 0
        trainable_params = 0
        
        for param in self.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        print(f"Encoder Total Parameters: {total_params:,}")
        print(f"Encoder Trainable Parameters: {trainable_params:,}")
        print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%")