"""
Adapter module implementations for various temporal architectures
Each adapter transforms input features to output embeddings without knowledge of the surrounding system
"""

import torch
import torch.nn as nn
import math

def init_weights(module):
    """Initialize the weights of a module using Xavier/Glorot initialization."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class SelfAttentionAdapter(nn.Module):
    """
    Simple multi-layer self-attention adapter for variable-length sequences.
    Assumes input is already padded and has positional encodings.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        num_layers,
        dropout,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(input_dim, output_dim)
        
        # Initialize weights
        self.apply(init_weights)
        
        # Initialize attention weights specifically
        for layer in self.transformer.layers:
            # Initialize self-attention weights
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            if layer.self_attn.in_proj_bias is not None:
                nn.init.zeros_(layer.self_attn.in_proj_bias)
            if layer.self_attn.out_proj.bias is not None:
                nn.init.zeros_(layer.self_attn.out_proj.bias)
            
            # Initialize feed-forward weights
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
            if layer.linear1.bias is not None:
                nn.init.zeros_(layer.linear1.bias)
            if layer.linear2.bias is not None:
                nn.init.zeros_(layer.linear2.bias)

    def forward(self, x, padding_mask=None):
        # x: [batch, seq_len, input_dim]
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.projection(x)
