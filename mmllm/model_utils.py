"""
Model utilities for SpeechBCI multimodal language models
"""

from peft import LoraConfig, get_peft_model, TaskType
from .model_adapters.transformer_adapter import TransformerAdapter

import torch.nn as nn
from .MMLLM import MMLLM


# Factory function to create appropriate model based on type
def create_embedding_model(
    model_type,
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
    Create two separate components: adapter and model
    """
    # Create appropriate adapter based on adapter_type
    if adapter_type == "attention":
        from mmllm.model_adapters.transformer_adapter import TransformerAdapter
        
        adapter = TransformerAdapter(
            embed_dim=embed_dim,
            output_dim=base_model.config.hidden_size,  # Note: parameter name may vary
            adapter_type=adapter_type,
            attention_mode=attention_mode,
            window_size=window_size,
            total_input_dim=total_input_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
    
    # Create MMLLM without adapter
    mmllm = MMLLM(base_model=base_model)
    
    # Return both components
    return mmllm, adapter

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