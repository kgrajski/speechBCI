import torch
import torch.nn as nn
from transformers import AutoModel

class MMLLM(nn.Module):
    """
    Multimodal Language Model that only accepts pre-embedded inputs
    """
    def __init__(self, base_model, **kwargs):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, inputs_embeds, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass using pre-embedded inputs
        """
        # Get base model outputs
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
            
        return outputs
        
    def generate(self, inputs_embeds, **kwargs):
        """
        Text generation using pre-embedded inputs
        """
        return self.base_model.generate(inputs_embeds=inputs_embeds, **kwargs)
    
    def print_trainable_parameters(self):
        """
        Print the number of trainable parameters in the model
        """
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Updated breakdown - no adapter inside MMLLM
        adapter_params = sum(p.numel() for n, p in self.named_parameters() 
                        if 'input_adapter' in n and p.requires_grad)
        other_params = trainable_params - adapter_params
        
        print(f"  - Adapter parameters: {adapter_params:,} ({100 * adapter_params / trainable_params:.2f}%)")
        print(f"  - Other parameters: {other_params:,} ({100 * other_params / trainable_params:.2f}%)")
