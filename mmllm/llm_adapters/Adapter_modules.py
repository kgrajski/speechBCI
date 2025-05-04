"""
Adapter module implementations for various temporal architectures
Each adapter transforms input features to output embeddings without knowledge of the surrounding system
"""

import torch
import torch.nn as nn

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

    def forward(self, x, padding_mask=None):
        # x: [batch, seq_len, input_dim]
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.projection(x)
