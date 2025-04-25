"""
Model utilities for SpeechBCI multimodal language models with compressed data support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
import gc


# Compressed Adapter Classes

class CompressedLinearAdapter(nn.Module):
    """
    Simple linear adapter for fixed-sized compressed tokens.
    Projects compressed token representations directly to model dimension.
    """
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, attention_mask=None):
        # x shape: [batch_size, num_tokens, token_dim]
        output = self.projection(x)
        output = self.dropout(output)
        return self.layer_norm(output)


class CompressedAttentionAdapter(nn.Module):
    """
    Attention-based adapter for compressed tokens.
    Applies self-attention across compressed tokens before projection.
    """
    
    def __init__(self, input_dim, output_dim, num_heads=8, dropout=0.1, 
                 mode="global", window_size=None):
        super().__init__()
        self.mode = mode
        self.window_size = window_size
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # First normalization
        residual = x
        x = self.layer_norm1(x)
        
        # Process attention mask for multihead attention if provided
        attn_mask = None
        if attention_mask is not None:
            # Convert boolean mask to float mask for attention
            attn_mask = attention_mask.float()
            attn_mask = attn_mask.masked_fill(
                attention_mask.logical_not(), float("-inf")
            )
        
        # Apply self-attention
        attn_output, _ = self.self_attention(
            query=x, 
            key=x, 
            value=x,
            attn_mask=attn_mask
        )
        
        # Residual connection after attention
        x = residual + attn_output
        
        # Second normalization and feed-forward
        residual = x
        x = self.layer_norm2(x)
        
        # Project to model dimension
        x = self.projection(x)
        x = self.dropout(x)
        
        return x


# Compressed Model Adapter Classes

class CompressedT5Adapter(nn.Module):
    """
    T5 adapter that supports compressed data input.
    
    This adapter takes fixed-size compressed token representations and
    adapts them for T5 language model processing.
    """
    
    def __init__(self, t5_model, embedding_dim, adapter_type="linear", custom_adapter=None):
        super().__init__()
        self.t5_model = t5_model
        self.embedding_dim = embedding_dim
        self.hidden_dim = t5_model.config.d_model  # T5 hidden dimension
        self.adapter_type = adapter_type.lower()
        
        # Use custom adapter if provided, otherwise build one
        if custom_adapter is not None:
            self.input_adapter = custom_adapter
        else:
            self.input_adapter = self._build_input_adapter()
    
    def _build_input_adapter(self):
        """Create appropriate adapter based on type."""
        if self.adapter_type == "linear":
            return CompressedLinearAdapter(self.embedding_dim, self.hidden_dim)
        elif self.adapter_type == "attention":
            return CompressedAttentionAdapter(
                self.embedding_dim, 
                self.hidden_dim,
                num_heads=8
            )
        else:
            raise ValueError(f"Unsupported compressed adapter type: {self.adapter_type}")
    
    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        """
        Forward pass through adapter and T5 model.
        
        Args:
            inputs_embeds: Compressed token embeddings [batch_size, num_tokens, embedding_dim]
            attention_mask: Optional attention mask for sequence
            labels: Optional target labels for computing loss
            
        Returns:
            T5 model outputs
        """
        # Process through adapter
        adapted_embeds = self.input_adapter(inputs_embeds, attention_mask)
        
        # Forward through T5
        outputs = self.t5_model(
            inputs_embeds=adapted_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate(self, inputs_embeds, attention_mask=None, **kwargs):
        """
        Generate text from compressed input embeddings.
        
        Args:
            inputs_embeds: Compressed token embeddings [batch_size, num_tokens, embedding_dim]
            attention_mask: Optional attention mask
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs
        """
        # Process through adapter
        adapted_embeds = self.input_adapter(inputs_embeds, attention_mask)
        
        # Generate with T5
        return self.t5_model.generate(
            inputs_embeds=adapted_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def print_trainable_parameters(self):
        """Print number of trainable parameters in the model."""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")


class CompressedBartAdapter(nn.Module):
    """
    BART adapter that supports compressed data input.
    
    This adapter takes fixed-size compressed token representations and
    adapts them for BART language model processing.
    """
    
    def __init__(self, bart_model, embedding_dim, adapter_type="linear", custom_adapter=None):
        super().__init__()
        self.bart_model = bart_model
        self.embedding_dim = embedding_dim
        self.hidden_dim = bart_model.config.d_model  # BART hidden dimension
        self.adapter_type = adapter_type.lower()
        
        # Use custom adapter if provided, otherwise build one
        if custom_adapter is not None:
            self.input_adapter = custom_adapter
        else:
            self.input_adapter = self._build_input_adapter()
    
    def _build_input_adapter(self):
        """Create appropriate adapter based on type."""
        if self.adapter_type == "linear":
            return CompressedLinearAdapter(self.embedding_dim, self.hidden_dim)
        elif self.adapter_type == "attention":
            return CompressedAttentionAdapter(
                self.embedding_dim, 
                self.hidden_dim,
                num_heads=8
            )
        else:
            raise ValueError(f"Unsupported compressed adapter type: {self.adapter_type}")
    
    def forward(self, inputs_embeds, attention_mask=None, labels=None, 
                decoder_input_ids=None, decoder_attention_mask=None, 
                decoder_inputs_embeds=None, **kwargs):
        """
        Forward pass through adapter and BART model.
        
        Args:
            inputs_embeds: Compressed token embeddings [batch_size, num_tokens, embedding_dim]
            attention_mask: Optional attention mask for sequence
            labels: Optional target labels for computing loss
            decoder_input_ids: Optional decoder input IDs
            decoder_attention_mask: Optional decoder attention mask
            decoder_inputs_embeds: Optional decoder input embeddings
            **kwargs: Additional model parameters
            
        Returns:
            BART model outputs
        """
        # Process through adapter
        adapted_embeds = self.input_adapter(inputs_embeds, attention_mask)
        
        # Auto-create decoder inputs if needed and labels aren't provided
        if labels is None and decoder_input_ids is None and decoder_inputs_embeds is None:
            batch_size = inputs_embeds.shape[0]
            decoder_input_ids = torch.ones(
                (batch_size, 1), 
                dtype=torch.long, 
                device=inputs_embeds.device
            ) * self.bart_model.config.decoder_start_token_id
            
            # Print diagnostic message during development
            #print("Auto-created decoder_input_ids for BART forward pass")
        
        # Forward through BART
        outputs = self.bart_model(
            inputs_embeds=adapted_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **kwargs
        )
        
        return outputs
    
    def generate(self, inputs_embeds, attention_mask=None, 
                decoder_input_ids=None, **kwargs):
        """
        Generate text from compressed input embeddings.
        
        Args:
            inputs_embeds: Compressed token embeddings [batch_size, num_tokens, embedding_dim]
            attention_mask: Optional attention mask
            decoder_input_ids: Optional decoder input IDs 
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs
        """
        # Process through adapter
        adapted_embeds = self.input_adapter(inputs_embeds, attention_mask)
        
        # Auto-create decoder inputs if needed (should rarely be necessary for generation)
        if decoder_input_ids is None and 'decoder_input_ids' not in kwargs:
            batch_size = inputs_embeds.shape[0]
            decoder_input_ids = torch.ones(
                (batch_size, 1),
                dtype=torch.long,
                device=inputs_embeds.device
            ) * self.bart_model.config.decoder_start_token_id
            
            # Print diagnostic message during development
            #print("Auto-created decoder_input_ids for BART generation")
        
        # Generate with BART
        return self.bart_model.generate(
            inputs_embeds=adapted_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
    
    def print_trainable_parameters(self):
        """Print number of trainable parameters in the model."""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")


# Model Creation Functions

def create_embedding_model(
    model_type,
    base_model,
    embedding_dim=64,
    adapter_type="linear",
    attention_mode="global",
    window_size=None,
    is_compressed=False
):
    """
    Factory function to create an appropriate adapter model.

    Args:
        model_type: Type of model ('t5', 'bart', etc.)
        base_model: Base language model
        embedding_dim: Dimension of input embeddings (VQVAE or compressed)
        adapter_type: Type of adapter architecture
        attention_mode: Type of attention pattern (for non-compressed only)
        window_size: Size of attention window (for non-compressed only)
        is_compressed: Whether using compressed data format

    Returns:
        Adapter model for the specified model type
    """
    model_type = model_type.lower()
    adapter_type = adapter_type.lower()
    
    # For compressed data, use specialized adapters
    if is_compressed:
        print(f"Using compressed data adapter: {adapter_type}")
        
        # Create compressed adapter model
        if model_type == "t5":
            return CompressedT5Adapter(base_model, embedding_dim, adapter_type)
        elif model_type == "bart":
            return CompressedBartAdapter(base_model, embedding_dim, adapter_type)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    else:
        # For standard VQVAE embeddings, import the original adapters
        print(f"Using standard adapter: {adapter_type}")
        try:
            from mmllm.llm_encoders import T5Adapter, BartAdapter
            
            if model_type == "t5":
                return T5Adapter(
                    base_model, 
                    embedding_dim, 
                    adapter_type,
                    attention_mode,
                    window_size
                )
            elif model_type == "bart":
                return BartAdapter(
                    base_model, 
                    embedding_dim, 
                    adapter_type,
                    attention_mode,
                    window_size
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except ImportError as e:
            print(f"Error importing original adapters: {e}")
            raise


def get_lora_model(base_model, model_type="t5", r=16, alpha=32, dropout=0.1):
    """
    Apply LoRA configuration to a language model.

    Args:
        base_model: Base language model
        model_type: Type of model ('t5', 'bart', etc.)
        r: LoRA rank parameter
        alpha: LoRA alpha scaling factor
        dropout: Dropout probability for LoRA layers

    Returns:
        PEFT model with LoRA applied
    """
    model_type = model_type.lower()

    # Configure target modules based on model type
    if model_type == "t5":
        target_modules = ["q", "v"]
    elif model_type == "bart":
        target_modules = ["q_proj", "v_proj"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Common LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    lora_model = get_peft_model(base_model, lora_config)
    return lora_model