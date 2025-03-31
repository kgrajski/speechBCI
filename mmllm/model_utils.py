"""
Model utilities for SpeechBCI multimodal language models
"""

import torch
from peft import LoraConfig, get_peft_model, TaskType
import gc
from mmllm.model_adapters import T5Adapter, BartAdapter
import torch.nn as nn


# Factory function to create appropriate model based on type
def create_embedding_model(
    model_type, 
    base_model, 
    embedding_dim=64, 
    adapter_type="linear",
    attention_mode="global",
    window_size=None
):
    """
    Factory function to create an appropriate adapter model based on model type.

    Args:
        model_type (str): Type of model ('t5', 'bart', etc.)
        base_model: The base language model
        embedding_dim (int): Dimension of input VQVAE embeddings
        adapter_type (str): Type of adapter architecture ('linear', 'lstm', 'conv', 'attention', 'rnn')
        attention_mode (str): Type of attention pattern ('global', 'causal', 'local')
        window_size (int): Size of attention window for local attention

    Returns:
        Adapter model for the specified model type
    """
    model_type = model_type.lower()

    if model_type == "t5":
        return T5Adapter(
            base_model, 
            embedding_dim, 
            adapter_type,
            attention_mode,
            window_size
        )
    elif model_type == "bart":
        return BartAdapter(
            base_model, 
            embedding_dim, 
            adapter_type,
            attention_mode,
            window_size
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Choose appropriate adapter based on adapter_type
    if adapter_type == 'linear':
        # Existing linear adapter code
        adapter = LinearAdapter(embedding_dim, hidden_dim)
    elif adapter_type == 'lstm':
        # Existing LSTM adapter code 
        adapter = LSTMAdapter(embedding_dim, hidden_dim)
    elif adapter_type == 'conv':
        # Existing convolutional adapter code
        adapter = ConvAdapter(embedding_dim, hidden_dim)
    elif adapter_type == 'attention':
        # Existing attention adapter code
        adapter = AttentionAdapter(embedding_dim, hidden_dim, attention_mode, window_size)
    elif adapter_type == 'rnn':
        # New RNN adapter
        adapter = RNNAdapter(embedding_dim, hidden_dim)
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")


class RNNAdapter(nn.Module):
    """
    Standard RNN adapter for processing embedding sequences.
    
    This adapter uses a simple RNN layer followed by a projection to transform
    input embeddings before passing them to the language model.
    """
    def __init__(self, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the RNN adapter.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        # Process through RNN
        output, _ = self.rnn(x)
        
        # Apply projection, normalization and dropout
        output = self.projection(output)
        output = self.norm(output)
        output = self.dropout(output)
        
        return output


def get_lora_model(base_model, model_type="t5", r=16, alpha=32, dropout=0.1):
    """
    Apply LoRA configuration to a language model.

    Args:
        base_model: The base language model
        model_type (str): Type of model ('t5', 'bart', etc.)
        r: LoRA rank parameter (controls adaptation capacity)
        alpha: LoRA alpha scaling factor
        dropout: Dropout probability for LoRA layers

    Returns:
        PEFT model with LoRA applied
    """
    model_type = model_type.lower()

    # Configure target modules based on model type
    if model_type == "t5":
        target_modules = ["q", "v", "k", "o"]
    elif model_type == "bart":
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Common LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="lora_only",
    )

    lora_model = get_peft_model(base_model, lora_config)
    return lora_model


# For backward compatibility
CustomEmbeddingT5 = T5Adapter
