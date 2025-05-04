"""
SpeechBCI Multimodal Language Model Package

This package provides utilities for training and evaluating
multimodal language models for speech BCI data.
"""

# Make key components available at the package level
from mmllm.utils.model_utils import create_adapter_encoder_decoder_model
from mmllm.utils.training_utils import run_exp, training, generation, log_metrics
from mmllm.utils.data_utils import get_vqvae_codebook_average, calculate_wer
from mmllm.MMLLM import MultimodalLLM

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
