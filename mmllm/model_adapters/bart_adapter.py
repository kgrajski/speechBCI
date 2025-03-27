"""
BART-specific adapter implementation
"""

import torch
from mmllm.model_adapters.base_adapter import BaseModelAdapter


class BartAdapter(BaseModelAdapter):
    """
    Adapter model that wraps BART to accept custom embeddings as input.
    """

    def __init__(self, bart_model, embedding_dim=64):
        """
        Initialize the adapter with a BART model and projection layer.

        Args:
            bart_model: Base BART model (standard or LoRA-adapted)
            embedding_dim: Dimension of input VQVAE embeddings
        """
        super().__init__(bart_model, embedding_dim)
        self.bart_model = bart_model
        self.input_adapter = self._build_input_adapter()

    def _build_input_adapter(self):
        """Improved adapter based on literature recommendations"""
        return torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.bart_model.config.d_model * 2),
            torch.nn.LayerNorm(self.bart_model.config.d_model * 2),  # Add normalization
            torch.nn.LeakyReLU(0.2),  # Replace ReLU with LeakyReLU
            torch.nn.Dropout(0.1),    # Add dropout for regularization
            torch.nn.Linear(
                self.bart_model.config.d_model * 2, self.bart_model.config.d_model
            ),
            torch.nn.LayerNorm(self.bart_model.config.d_model),  # Final normalization
        )

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
