"""
SpeechBCI Multimodal Language Model Package

This package provides utilities for training and evaluating
multimodal language models for speech BCI data.
"""

# Make key components available at the package level
from mmllm.model_utils import get_lora_model
from mmllm.training_utils import run_exp, train_epoch, evaluate
from mmllm.data_utils import get_vqvae_codebook_average, calculate_wer

__all__ = [
    "CustomEmbeddingT5",
    "get_lora_model",
    "run_exp",
    "train_epoch",
    "evaluate",
    "get_vqvae_codebook_average",
    "calculate_wer",
]
