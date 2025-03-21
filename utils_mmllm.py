"""
Multimodal Language Model Utilities for Speech BCI

This module provides utilities for training and evaluating multimodal language models
that process Speech BCI data. It includes custom model adapters, training loops,
evaluation functions, and memory management utilities.

NOTE: This file now serves as a compatibility layer importing from the mmllm package.
For new code, import directly from the mmllm package modules.
"""

import os
import numpy as np
import torch
import gc

# GPU memory management configuration
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Re-export all components from submodules
from mmllm.model_utils import CustomEmbeddingT5, get_lora_model
from mmllm.training_utils import run_exp, train_epoch, evaluate
from mmllm.data_utils import get_vqvae_codebook_average, calculate_wer

# Define what should be imported when using "from utils_mmllm import *"
__all__ = [
    "CustomEmbeddingT5",
    "get_lora_model",
    "run_exp",
    "train_epoch",
    "evaluate",
    "calculate_wer",
    "get_vqvae_codebook_average",
]
