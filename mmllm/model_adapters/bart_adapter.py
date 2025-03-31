"""
BART-specific adapter implementation
"""

import torch
from torch import nn
from mmllm.model_adapters.base_adapter import BaseModelAdapter
from mmllm.model_adapters.adapter_modules import (
    LSTMAdapter, RNNAdapter,
    ConvolutionalAdapter as ConvAdapter,  # Add alias
    SelfAttentionAdapter as AttentionAdapter  # Add alias
)

class BartAdapter(BaseModelAdapter):
    """
    Adapter model that wraps BART to accept custom embeddings as input.
    """

    def __init__(self, bart_model, embedding_dim=64, adapter_type="linear", attention_mode="global", window_size=None):
        """
        Initialize the adapter with a BART model and projection layer.

        Args:
            bart_model: Base BART model (standard or LoRA-adapted)
            embedding_dim: Dimension of input VQVAE embeddings
            adapter_type: Type of adapter architecture ('linear', 'lstm', 'conv', 'attention', 'rnn')
            attention_mode: Type of attention pattern ('global', 'causal', 'local')
            window_size: Size of attention window for local attention
        """
        super().__init__(bart_model, embedding_dim, adapter_type, attention_mode, window_size)
        self.bart_model = bart_model
        self.input_adapter = self._build_input_adapter()

    def _build_input_adapter(self):
        """Build the adapter that processes input embeddings."""
        if self.adapter_type == "linear":
            return torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, self.bart_model.config.d_model * 2),
                torch.nn.LayerNorm(self.bart_model.config.d_model * 2),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(
                    self.bart_model.config.d_model * 2, self.bart_model.config.d_model
                ),
                torch.nn.LayerNorm(self.bart_model.config.d_model),
            )
        elif self.adapter_type == "lstm":
            return LSTMAdapter(self.embedding_dim, self.bart_model.config.d_model)
        elif self.adapter_type == "conv":
            return ConvAdapter(self.embedding_dim, self.bart_model.config.d_model)
        elif self.adapter_type == "attention":
            return AttentionAdapter(self.embedding_dim, self.bart_model.config.d_model)
        elif self.adapter_type == "rnn":
            return RNNAdapter(self.embedding_dim, self.bart_model.config.d_model)
        else:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")

    def forward(self, inputs_embeds, attention_mask, labels=None):
        """
        Forward pass through the adapter and BART model.

        Args:
            inputs_embeds: VQVAE embeddings [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask for input sequence
            labels: Optional target labels for computing loss

        Returns:
            BART model outputs
        """
        # Ensure inputs are fresh tensors
        inputs_embeds = inputs_embeds.detach()
        attention_mask = attention_mask.detach()
        if labels is not None:
            labels = labels.detach()

        # Map custom embeddings to BART embedding dimension
        adapted_embeds = self.input_adapter(inputs_embeds)

        # Forward pass through BART model with our adapted embeddings
        outputs = self.bart_model(
            inputs_embeds=adapted_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs

    def generate(self, inputs_embeds, attention_mask=None, **kwargs):
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

        # Map custom embeddings to BART embedding dimension
        adapted_embeds = self.input_adapter(inputs_embeds)

        # Generate with BART model
        return self.bart_model.generate(
            inputs_embeds=adapted_embeds, attention_mask=attention_mask, **kwargs
        )

    def _create_attention_mask(self, seq_len, attention_mode, window_size=None):
        # Create mask as before, but convert boolean mask to float mask with -inf
        if attention_mode == "causal":
            # Create causal mask (True means masked position)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            # Convert to float mask with -inf for masked positions
            float_mask = torch.zeros_like(mask, dtype=torch.float)
            float_mask.masked_fill_(mask, float('-inf'))
            return float_mask
        # Similar changes for other mask types
