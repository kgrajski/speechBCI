"""
Model adapter implementations for different language models
"""

# Import from deprecated modules if still needed during transition
from ..deprecated.base_adapter import BaseModelAdapter
from mmllm.deprecated.t5_adapter import T5Adapter
from mmllm.deprecated.bart_adapter import BartAdapter

# Import current modules
from mmllm.model_adapters.adapter_modules import (
    LSTMAdapter, 
    ConvolutionalAdapter, 
    SelfAttentionAdapter,
    RNNAdapter
)
from mmllm.model_adapters.transformer_adapter import TransformerAdapter

__all__ = [
    # Deprecated adapters
    "BaseModelAdapter",
    "T5Adapter",
    "BartAdapter",
    
    # Current adapter modules
    "LSTMAdapter",
    "ConvolutionalAdapter",
    "SelfAttentionAdapter",
    "RNNAdapter",
    
    # Unified adapter
    "TransformerAdapter"
]
