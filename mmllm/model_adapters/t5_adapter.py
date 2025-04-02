"""
T5-specific adapter implementation
"""

import torch
from torch import nn
from mmllm.model_adapters.base_adapter import BaseModelAdapter
from mmllm.model_adapters.adapter_modules import (
    LinearAdapter, LSTMAdapter, RNNAdapter,
    ConvolutionalAdapter as ConvAdapter,  # Add alias
    SelfAttentionAdapter as AttentionAdapter  # Add alias
)


class T5Adapter(BaseModelAdapter):
    """
    Adapter model that wraps T5 to accept custom embeddings as input.
    """

    def __init__(self, t5_model, embedding_dim=64, adapter_type="linear", attention_mode="global", window_size=None):
        """
        Initialize the adapter with a T5 model and projection layer.

        Args:
            t5_model: Base T5 model (standard or LoRA-adapted)
            embedding_dim: Dimension of input VQVAE embeddings
            adapter_type: Type of adapter architecture ('linear', 'lstm', 'conv', 'attention', 'rnn')
            attention_mode: Type of attention pattern ('global', 'causal', 'local')
            window_size: Size of attention window for local attention
        """
        super().__init__(t5_model, embedding_dim, adapter_type, attention_mode, window_size)
        self.t5_model = t5_model
        self.input_adapter = self._build_input_adapter()

    def _build_input_adapter(self):
        """Build the adapter that processes input embeddings."""
        if self.adapter_type == "linear":
            return LinearAdapter(self.embedding_dim, self.hidden_dim)
        elif self.adapter_type == "lstm":
            return LSTMAdapter(self.embedding_dim, self.hidden_dim)
        elif self.adapter_type == "conv":
            return ConvAdapter(self.embedding_dim, self.hidden_dim)
        elif self.adapter_type == "attention":
            return AttentionAdapter(
                self.embedding_dim, 
                self.hidden_dim, 
                self.attention_mode, 
                self.window_size
            )
        elif self.adapter_type == "rnn":
            return RNNAdapter(self.embedding_dim, self.hidden_dim)
        else:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")

    def forward(self, inputs_embeds, attention_mask, labels=None):
        """
        Forward pass through the adapter and T5 model.

        Args:
            inputs_embeds: VQVAE embeddings [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask for input sequence
            decoder_input_ids: Optional decoder input IDs for teacher forcing
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
        adapted_embeds = self.input_adapter(inputs_embeds)

        # Forward pass through T5 model with our adapted embeddings

        outputs = self.t5_model(
            inputs_embeds=adapted_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs

    def generate(self, inputs_embeds, attention_mask, **kwargs):
        """
        Generate text from input embeddings.

        Args:
            inputs_embeds: VQVAE embeddings [batch_size, seq_len, embedding_dim]
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
        adapted_embeds = self.input_adapter(inputs_embeds)

        # Generate with T5 model
        return self.t5_model.generate(
            inputs_embeds=adapted_embeds, attention_mask=attention_mask, **kwargs
        )
