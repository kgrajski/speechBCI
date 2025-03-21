"""
Base adapter class for all language model adapters
"""

import torch
import gc


class BaseModelAdapter(torch.nn.Module):
    """
    Base adapter class that defines the interface for all model adapters
    """

    def __init__(self, base_model, embedding_dim=64):
        """
        Initialize the base adapter with a model.

        Args:
            base_model: Base language model
            embedding_dim: Dimension of input VQVAE embeddings
        """
        super().__init__()
        self.base_model = base_model
        self.embedding_dim = embedding_dim

    def _build_input_adapter(self):
        """
        Build the input adapter that maps VQVAE embeddings to model dimensions.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _build_input_adapter")

    def forward(
        self, inputs_embeds, attention_mask, decoder_input_ids=None, labels=None
    ):
        """
        Forward pass through the adapter and model.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward")

    def generate(self, inputs_embeds, attention_mask, **kwargs):
        """
        Generate text from input embeddings.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate")

    def print_trainable_parameters(self):
        """
        Print information about trainable parameters.
        """
        if hasattr(self.base_model, "print_trainable_parameters"):
            return self.base_model.print_trainable_parameters()
        else:
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            all_params = sum(p.numel() for p in self.parameters())
            print(f"Trainable parameters: {trainable_params}")
            print(f"Total parameters: {all_params}")
            print(f"Trainable%: {100 * trainable_params / all_params:.2f}%")
