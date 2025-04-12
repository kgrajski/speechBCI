"""
Model utilities for SpeechBCI multimodal language models
"""

from peft import LoraConfig, get_peft_model, TaskType
from mmllm.model_adapters.transformer_adapter import TransformerAdapter
import torch
import torch.nn as nn
from .deprecated.MMLLM import MMLLM
from .MultimodalLLM import MultimodalLLM

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
)


# Factory function to create appropriate model based on type
def create_embedding_model(
    adapter_type,
    base_model_type,
    input_dim,
    attention_mode,
    window_size,
    num_heads,
    num_layers,
    dropout,
):

    # Create an instance of the base model
    if base_model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
        base_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
    elif base_model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        # Add sentence boundary tokens
        special_tokens = {"additional_special_tokens": ["<sentence>", "</sentence>"]}
        tokenizer.add_special_tokens(special_tokens)
        base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        # Resize model embeddings to match updated tokenizer
        base_model.resize_token_embeddings(len(tokenizer))
        print("Using standard BART without multilingual support")
        
    else:
        raise ValueError(f"Unsupported model type: {base_model_type}")
    
    # Create appropriate adapter based on adapter_type
    if adapter_type == "attention":
        adapter = TransformerAdapter(
            input_dim=input_dim,
            output_dim=base_model.config.hidden_size,
            adapter_type=adapter_type,
            attention_mode=attention_mode,
            window_size=window_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")

    # Create an instance of the adapter + base_model
    mmllm = MultimodalLLM(
        adapter_type=adapter_type,
        model_type=base_model_type,
        input_adapter=adapter,
        base_model=base_model,
        tokenizer=tokenizer,
    )

    # Freeze the base model parameters
    for param in mmllm.base_model.parameters():
        param.requires_grad = False

    # Return the
    return mmllm


# Deprecated for now, but it around in case of base model training, after all.
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
