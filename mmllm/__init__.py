"""
SpeechBCI Multimodal Language Model Package

This package provides utilities for training and evaluating
multimodal language models for speech BCI data.
"""

# Make key components available at the package level
from mmllm.model_utils import create_embedding_model, get_lora_model
from mmllm.training_utils import run_exp, training, generation, log_metrics
from mmllm.data_utils import get_vqvae_codebook_average, calculate_wer
from mmllm.MultimodalLLM import MultimodalLLM

__all__ = [
    "create_embedding_model",
    "get_lora_model",
    "run_exp",
    "training",
    "generation",
    "log_metrics",
    "get_vqvae_codebook_average",
    "calculate_wer",
    "MultimodalLLM",
]
