"""
Model utilities for SpeechBCI multimodal language models
"""

from peft import LoraConfig, get_peft_model, TaskType
from .model_adapters.transformer_adapter import TransformerAdapter
import torch.nn as nn


# Factory function to create appropriate model based on type
def create_embedding_model(
    model_type, 
    base_model, 
    embed_dim,    # Per-timestep feature dimensio
    adapter_type,
    attention_mode,
    window_size,
    total_input_dim,  # Total flattened input dimension (linear adapter only)
    num_heads,        # New parameters
    num_layers,
    dropout,
):
    """
    Factory function to create an appropriate adapter model.

    Args:
        model_type (str): Type of model ('t5', 'bart', etc.)
        base_model: The base language model
        embed_dim (int): Dimension of features at each timestep
        adapter_type (str): Type of adapter architecture
        attention_mode (str): Type of attention pattern
        window_size (int): Size of attention window for local attention
        seq_length (int, optional): Maximum sequence length
        total_input_dim (int, optional): Total flattened dimension for linear adapter
        num_heads (int): Number of attention heads
        num_layers (int): Number of layers in the adapter
        dropout (float): Dropout rate for the adapter

    Returns:
        TransformerAdapter: The adapter model for the specified model type
    """
    
    # Validate the model type
    supported_models = ['t5', 'bart']
    model_type = model_type.lower()
    if model_type not in supported_models:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {supported_models}")
    
    # Create adapter with appropriate parameters
    return TransformerAdapter(
        base_model=base_model,
        embed_dim=embed_dim,
        adapter_type=adapter_type,
        attention_mode=attention_mode,
        window_size=window_size,
        total_input_dim=total_input_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )

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