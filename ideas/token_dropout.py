def apply_token_dropout(self, token_indices, quantized_tokens, p=0.1):
    """Apply dropout to force model to use diverse tokens"""
    if not self.training:
        return token_indices, quantized_tokens
        
    batch_size, seq_len = token_indices.shape
    # Get device from input
    device = token_indices.device
    
    # Create dropout mask
    mask = torch.rand(batch_size, seq_len, device=device) > p
    
    # For indices we want to drop, replace with random indices
    random_indices = torch.randint(0, self.output_tokens, 
                                  (batch_size, seq_len), 
                                  device=device)
    # Only replace where mask is False
    token_indices = torch.where(mask, token_indices, random_indices)
    
    # The most important fix - create a local copy of codebook on the right device
    codebook = self.codebook.to(device)
    
    # Update quantized tokens to match new indices
    flat_indices = token_indices.reshape(-1)
    quantized = codebook[flat_indices]
    quantized = quantized.reshape(batch_size, seq_len, -1)
    
    return token_indices, quantized