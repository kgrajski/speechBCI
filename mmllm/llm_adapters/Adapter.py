"""
Unified Adapter for transformer-based models (T5, BART)
"""

import torch
from torch import nn
from mmllm.llm_adapters.Adapter_modules import (
    SelfAttentionAdapter as AttentionAdapter,
)


class Adapter(nn.Module):
    """
    Universal Adapter for transformer-based models (e.g., BART, T5).

    This adapter transforms input features (such as those from a VQ-VAE or other multi-modal encoder)
    into a sequence of embeddings suitable for transformer-based encoder-decoder models.
    It supports attention-based adapters and projects input features to the target output dimension,
    adding any necessary attention mechanisms.

    Args:
        input_dim: Dimension of features at each timestep.
        output_dim: Target output dimension for features to be passed to the LLM encoder.
        adapter_type: Type of adapter to use ('attention' supported).
        attention_mode: Attention pattern ('global', 'causal', 'local').
        window_size: Size of attention window for local attention.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        adapter_type,
        num_heads,
        num_layers,
        dropout,
    ):
        super().__init__()

        # Store dimensions
        self.input_dim = input_dim # Of the input features (non time dimension)
        self.output_dim = output_dim  # Adapter output which is LLM encoder input dimension
        self.adapter_type = adapter_type.lower()

        # Configure the adapter based on type
        if self.adapter_type == "attention":
            self.adapter = AttentionAdapter(
                input_dim=input_dim,
                output_dim=output_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")

    def forward(self, inputs, padding_masks=None):
        """
        Forward pass through the adapter only.

        Args:
            inputs: Input features with shape:
                - For sequence models: [batch_size, seq_len, input_dim]
                - For linear encoder: [batch_size, total_input_dim]
            padding_masks: Optional mask indicating which positions are real data (1) vs padding (0)

        Returns:
            Tensor with shape [batch_size, seq_len, output_dim]
        """
        # Transform input features to output dimensions
        return self.adapter(inputs, padding_masks)
