"""
Unified adapter for transformer-based models (T5, BART)
"""

import torch
from torch import nn
from mmllm.model_adapters.adapter_modules import (
    LSTMAdapter, RNNAdapter,
    ConvolutionalAdapter as ConvAdapter,
    SelfAttentionAdapter as AttentionAdapter
)

class TransformerAdapter(nn.Module):
    """
    Universal adapter for transformer models with similar API (T5, BART)
    """

    def __init__(
        self, 
        base_model, 
        embed_dim,
        adapter_type,
        attention_mode,
        window_size,
        total_input_dim,
        num_heads,
        num_layers,
        dropout,
    ):
        """
        Initialize the adapter for any transformer model.

        Args:
            base_model: Base model (T5, BART, etc.)
            embed_dim: Dimension of features at each timestep
            adapter_type: Type of adapter ('linear', 'lstm', 'conv', 'attention', 'rnn')
            attention_mode: Attention pattern ('global', 'causal', 'local')
            window_size: Size of attention window for local attention
            seq_length: Maximum sequence length for context
            total_input_dim: Total flattened dimension for linear adapter
        """
        super().__init__()
        self.base_model = base_model
        self.adapter_type = adapter_type.lower()
        self.model_dim = self.base_model.config.d_model  # Works for both T5 and BART
        
        # Determine if we're using a sequence model
        self.is_sequence_model = self.adapter_type in ["lstm", "rnn", "conv", "attention"]
        
        # Determine input dimension based on adapter type
        input_dim = total_input_dim if (self.adapter_type == "linear" and total_input_dim) else embed_dim
        
        # Configure the adapter based on type
        if self.adapter_type == "linear":
            # For linear adapters, input_dim represents the total flattened dimension
            self.adapter = nn.Linear(input_dim, self.model_dim)
        elif self.adapter_type == "lstm":
            # For sequence models, input_dim represents per-timestep features
            self.adapter = LSTMAdapter(input_dim, self.model_dim)
        elif self.adapter_type == "conv":
            self.adapter = ConvAdapter(input_dim, self.model_dim)
        elif self.adapter_type == "attention":
            self.adapter = AttentionAdapter(
                input_dim, 
                self.model_dim, 
                attention_mode, 
                window_size,
                num_heads,
                num_layers,
                dropout,
            )
        elif self.adapter_type == "rnn":
            self.adapter = RNNAdapter(input_dim, self.model_dim)
        else:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")

    def forward(self, inputs_embeds, attention_mask, labels=None):
        """
        Forward pass through the adapter and model.

        Args:
            inputs_embeds: Input features with shape:
                - For sequence models: [batch_size, seq_len, embed_dim]
                - For linear adapter: [batch_size, total_input_dim]
            attention_mask: Attention mask for input sequence
            labels: Optional target labels for computing loss

        Returns:
            Model outputs
        """
        # Ensure inputs are fresh tensors
        inputs_embeds = inputs_embeds.detach()
        attention_mask = attention_mask.detach()
        if labels is not None:
            labels = labels.detach()

        # Map custom embeddings to model embedding dimension
        adapted_embeds = self.adapter(inputs_embeds)

        # Forward pass through model with adapted embeddings
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
            inputs_embeds: Input features
            attention_mask: Attention mask for input sequence
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs
        """
        # Ensure inputs are fresh tensors
        inputs_embeds = inputs_embeds.detach()
        if attention_mask is not None:
            attention_mask = attention_mask.detach()

        # Map custom embeddings to model embedding dimension
        adapted_embeds = self.adapter(inputs_embeds)

        # Generate text
        return self.base_model.generate(
            inputs_embeds=adapted_embeds, 
            attention_mask=attention_mask, 
            **kwargs
        )
        
    def print_trainable_parameters(self):
        """Print the number of trainable parameters in the model."""
        total_params = 0
        trainable_params = 0
        
        for param in self.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%")