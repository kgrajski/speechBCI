"""
Model utilities for SpeechBCI multimodal language models
"""

import torch
import torch.nn as nn

from mmllm.llm_adapters.Adapter import Adapter
from mmllm.MMLLM import MultimodalLLM
from transformers import (BartTokenizer, BartForConditionalGeneration)


# Factory function to create appropriate model based on type
def create_adapter_encoder_decoder_model(
    adapter_type,
    base_model_type,
    input_dim,
    num_heads,
    num_layers,
    dropout,
    diversity_loss_weight,
    encoder_reg_weight,
):

    # Create an instance of the base model
    if base_model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        # Add sentence boundary tokens
        special_tokens = {"additional_special_tokens": ["<sentence>", "</sentence>"]}
        tokenizer.add_special_tokens(special_tokens)
        base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        # Resize model embeddings to match updated tokenizer
        base_model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"Unsupported model type: {base_model_type}")

    # Test base model with a simple prompt
    print("\nTesting base model with simple prompt...")
    test_prompt = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_prompt, return_tensors="pt")
    outputs = base_model.generate(**inputs, max_length=50)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input prompt: {test_prompt}")
    print(f"Model output: {decoded_output}")

    # Create appropriate adapter based on adapter_type
    if adapter_type == "attention":
        input_adapter = Adapter(
            input_dim=input_dim,
            output_dim=base_model.config.hidden_size,
            adapter_type=adapter_type,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported encoder type: {adapter_type}")

    # Create an instance of the encoder + base_model
    mmllm = MultimodalLLM(
        adapter_type,
        base_model_type,
        input_adapter,
        base_model,
        tokenizer,
        diversity_loss_weight,
        encoder_reg_weight,
    )

    # Freeze the base model parameters
    for param in mmllm.base_model.parameters():
        param.requires_grad = False

    # Return the
    return mmllm