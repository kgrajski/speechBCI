"""
T5-specific adapter implementation
"""

import torch
from torch import nn
from mmllm.deprecated.base_adapter import BaseModelAdapter
from mmllm.model_adapters.adapter_modules import (
    LSTMAdapter, RNNAdapter,
    ConvolutionalAdapter as ConvAdapter,  # Add alias
    SelfAttentionAdapter as AttentionAdapter  # Add alias
)


class T5Adapter(nn.Module):
    """
    Adapter model that wraps T5 to accept custom embeddings as input.
    """

    def __init__(
        self, 
        base_model, 
        feature_dim,  # Renamed from embedding_dim for clarity
        adapter_type="linear",
        attention_mode="global",
        window_size=None,
        seq_length=None
    ):
        """
        Initialize the adapter with a T5 model and projection layer.

        Args:
            base_model: Base T5 model (standard or LoRA-adapted)
            feature_dim: Dimension of input VQVAE embeddings
            adapter_type: Type of adapter architecture ('linear', 'lstm', 'conv', 'attention', 'rnn')
            attention_mode: Type of attention pattern ('global', 'causal', 'local')
            window_size: Size of attention window for local attention
            seq_length: Sequence length for sequence models
        """
        super().__init__()
        self.base_model = base_model
        self.adapter_type = adapter_type.lower()
        
        # Determine if we're using a sequence model
        self.is_sequence_model = self.adapter_type in ["lstm", "rnn", "conv", "attention"]
        
        # Configure the adapter based on type
        if self.adapter_type == "linear":
            # For linear adapters, feature_dim represents the total flattened dimension
            self.adapter = nn.Linear(feature_dim, self.base_model.config.d_model)
        elif self.adapter_type == "lstm":
            # For sequence models, feature_dim represents per-timestep features
            self.adapter = LSTMAdapter(feature_dim, self.base_model.config.d_model)
        elif self.adapter_type == "conv":
            self.adapter = ConvAdapter(feature_dim, self.base_model.config.d_model)
        elif self.adapter_type == "attention":
            self.adapter = AttentionAdapter(
                feature_dim, 
                self.base_model.config.d_model, 
                attention_mode, 
                window_size
            )
        elif self.adapter_type == "rnn":
            self.adapter = RNNAdapter(feature_dim, self.base_model.config.d_model)
        else:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")

    def forward(self, inputs_embeds, attention_mask, labels=None):
        """
        Forward pass through the adapter and T5 model.

        Args:
            inputs_embeds: Input features with shape:
                - For sequence models: [batch_size, seq_len, feature_dim]
                - For linear adapter: [batch_size, total_input_dim]
            attention_mask: Attention mask for input sequence
            labels: Optional target labels for computing loss

        Returns:
            T5 model outputs
        """
        # Ensure inputs are fresh tensors
        inputs_embeds = inputs_embeds.detach()
        attention_mask = attention_mask.detach()
        if labels is not None:
            labels = labels.detach()

        # Map custom embeddings to T5 embedding dimension
        adapted_embeds = self.adapter(inputs_embeds)

        # Forward pass through T5 model with our adapted embeddings
        outputs = self.base_model(
            inputs_embeds=adapted_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs

    def generate(self, inputs_embeds, attention_mask, **kwargs):
        """
        Generate text from input embeddings.

        Args:
            inputs_embeds: VQVAE embeddings [batch_size, seq_len, feature_dim]
            attention_mask: Attention mask for input sequence
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs
        """
        # Ensure inputs are fresh tensors
        inputs_embeds = inputs_embeds.detach()
        if attention_mask is not None:
            attention_mask = attention_mask.detach()

        # Map custom embeddings to T5 embedding dimension
        adapted_embeds = self.adapter(inputs_embeds)

        # Generate with T5 model
        return self.base_model.generate(
            inputs_embeds=adapted_embeds, attention_mask=attention_mask, **kwargs
        )
