def create_embedding_model(
    model_type, 
    base_model, 
    embedding_dim,
    adapter_type,
    attention_mode="global",
    window_size=None,
    is_compressed=False  # New flag
):
    """Create a multimodal LLM with appropriate adapter."""
    
    # Determine model-specific dimensions
    if model_type == "t5":
        model_dim = 512  # t5-small encoder dim
    elif model_type == "bart":
        model_dim = 768  # bart-base encoder dim
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # For compressed data, we use a simplified adapter approach
    if is_compressed:
        if adapter_type == "linear":
            # Simple projection from compressed tokens to model dimension
            adapter = CompressedLinearAdapter(
                input_dim=embedding_dim, 
                output_dim=model_dim
            )
        elif adapter_type == "attention":
            # Self-attention on compressed tokens then projection
            adapter = CompressedAttentionAdapter(
                input_dim=embedding_dim,
                hidden_dim=model_dim,
                num_heads=8,
                mode=attention_mode,
                window_size=window_size
            )
        else:
            raise ValueError(f"Unsupported adapter type for compressed data: {adapter_type}")
    else:
        # Original adapter selection for VQ-VAE embeddings
        # ...existing code...
        
    # Create model with selected adapter
    if model_type == "t5":
        return T5WithEmbeddingAdapter(base_model, adapter)
    elif model_type == "bart":
        return BartWithEmbeddingAdapter(base_model, adapter)