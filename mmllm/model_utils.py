"""
Model utilities for SpeechBCI multimodal language models
"""

import torch
from peft import LoraConfig, get_peft_model, TaskType
import gc
from mmllm.model_adapters import T5Adapter, BartAdapter


# Factory function to create appropriate model based on type
def create_embedding_model(model_type, base_model, embedding_dim=64):
    """
    Factory function to create an appropriate adapter model based on model type.

    Args:
        model_type (str): Type of model ('t5', 'bart', etc.)
        base_model: The base language model
        embedding_dim (int): Dimension of input VQVAE embeddings

    Returns:
        Adapter model for the specified model type
    """
    model_type = model_type.lower()

    if model_type == "t5":
        return T5Adapter(base_model, embedding_dim)
    elif model_type == "bart":
        return BartAdapter(base_model, embedding_dim)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


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
