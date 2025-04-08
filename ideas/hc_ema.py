def initialize_codebook(self):
    """Initialize a codebook for vector quantization."""
    self.register_parameter(
        "codebook", 
        nn.Parameter(torch.randn(self.output_tokens, self.output_dim) * 0.02)
    )
    # Don't register buffers here - we'll create them on first use

def update_codebook_with_ema(self, encodings, flat_inputs, decay=0.99):
    """Safely update codebook with EMA, handling device issues"""
    if not self.training:
        return
    
    # Get device from encodings
    device = encodings.device
    
    # Create buffers on demand on the correct device
    if not hasattr(self, "ema_count") or self.ema_count.device != device:
        self.ema_count = torch.zeros(self.output_tokens, device=device)
        self.ema_weight = torch.zeros(self.output_tokens, self.output_dim, device=device)
    
    # Create one-hot encodings directly on the correct device
    encodings_onehot = torch.zeros(
        encodings.shape[0], self.output_tokens, 
        device=device, dtype=flat_inputs.dtype
    )
    encodings_onehot.scatter_(1, encodings.unsqueeze(1), 1)
    
    # Compute usage
    usage = encodings_onehot.sum(0)
    
    # Update EMA count
    self.ema_count = self.ema_count * decay + (1 - decay) * usage
    
    # Update EMA weight
    dw = torch.matmul(encodings_onehot.t(), flat_inputs)
    self.ema_weight = self.ema_weight * decay + (1 - decay) * dw
    
    # Update codebook - make sure it's on the right device
    codebook_updated = self.ema_weight / (self.ema_count.unsqueeze(-1) + 1e-5)
    self.codebook.data = codebook_updated.to(self.codebook.device)