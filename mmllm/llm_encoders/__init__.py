"""
Model adapter implementations for different language models
"""


# Import current modules
from mmllm.llm_encoders.encoder_modules import (
    LSTMEncoder, 
    ConvolutionalEncoder, 
    SelfAttentionEncoder,
    RNNEncoder
)
from mmllm.llm_encoders.transformer_encoder import TransformerEncoder

__all__ = [
    # Deprecated adapters
    "BaseModelEncoder",
    "T5Encoder",
    "BartEncoder",
    
    # Current adapter modules
    "LSTMEncoder",
    "ConvolutionalEncoder",
    "SelfAttentionEncoder",
    "RNNEncoder",
    
    # Unified adapter
    "TransformerEncoder"
]
