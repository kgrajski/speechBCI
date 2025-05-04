import os
import torch
import torch.nn as nn


class MultimodalLLM(nn.Module):
    """
    Base class for multimodal language models with various adapters.
    Note the design principle of clean separatetion of model and adapter.
    Operates on one input batch at a time.
    """

    def __init__(
            self,
            adapter_type,
            model_type,
            input_adapter,
            base_model,
            tokenizer,
            diversity_loss_weight,
            encoder_reg_weight,
        ):
        super().__init__()
        self.adapter_type = adapter_type
        self.model_type = model_type
        self.input_adapter = input_adapter
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.diversity_loss_weight = diversity_loss_weight
        self.encoder_reg_weight = encoder_reg_weight

    def forward(
        self,
        input_embeddings,
        padding_masks,
        positional_encodings,
        labels,
        **kwargs,
    ):

        # Combine positional encodings with input embeddings
        input_embeddings = input_embeddings + positional_encodings

        # Process the input embeddings through the adapter
        adapter_outputs = self.input_adapter(input_embeddings, padding_masks)

        # Forward through base model with updated padding masks
        mmllm_outputs = self.base_model(
            inputs_embeds=adapter_outputs,
            attention_mask=padding_masks,
            labels=labels,
            **kwargs,
        )
        
        # Apply updated padding mask to encoder outputs for loss computation
        masked_outputs = adapter_outputs * padding_masks.unsqueeze(-1)
        
        # Compute diversity loss on masked outputs
        diversity_loss = -torch.var(masked_outputs).mean() * self.diversity_loss_weight
        
        # Compute regularization loss on masked outputs
        reg_loss = torch.norm(masked_outputs) * self.encoder_reg_weight
        
        # Compute total loss
        total_loss = mmllm_outputs.loss + diversity_loss + reg_loss

        # Return both encoder outputs and all losses
        return adapter_outputs, {
            "loss": total_loss,
            "main_loss": mmllm_outputs.loss,
            "diversity_loss": diversity_loss,
            "reg_loss": reg_loss,
        }

    def generate(
        self,
        input_embeddings,
        padding_masks,
        positional_encodings,
        **kwargs,
    ):
        """
        Generate text from input embeddings.

        Args:
            input_embeddings: Input embeddings from VQ-VAE
            padding_masks: Mask indicating which positions are real data (1) vs padding (0)
            **kwargs: Additional arguments passed to the base model
        """
        # Combine positional encodings with input embeddings
        input_embeddings = input_embeddings + positional_encodings

        # Process the input embeddings through the adapter
        adapter_outputs = self.input_adapter(input_embeddings, padding_masks)

        # Add task-specific prompt if provided
        tokenizer = kwargs.pop("tokenizer", None)
        prompt_tokens = tokenizer(
            kwargs.pop("task_prompt", None), return_tensors="pt"
        ).input_ids.to(adapter_outputs.device)
        prompt_embeds = self.base_model.get_input_embeddings()(prompt_tokens)

        # Repeat prompt for each item in batch
        batch_size = adapter_outputs.shape[0]
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)

        # Concatenate with adapter outputs
        combined_embeds = torch.cat([prompt_embeds, adapter_outputs], dim=1)

        # Update padding mask if provided
        if padding_masks is not None:
            prompt_mask = torch.ones(
                batch_size, prompt_tokens.shape[1], device=padding_masks.device
            ).to(padding_masks.device)
            padding_masks = torch.cat([prompt_mask, padding_masks], dim=1)

        # Generate with processed embeddings
        return self.base_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=padding_masks,
            **kwargs,
        )

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
        print(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

        # Updated breakdown - no encoder inside MMLLM
        encoder_params = sum(
            p.numel()
            for n, p in self.named_parameters()
            if "input_adapter" in n and p.requires_grad
        )
        other_params = trainable_params - encoder_params

        print(
            f"  - encoder parameters: {encoder_params:,} ({100 * encoder_params / trainable_params:.2f}%)"
        )
        print(
            f"  - Other parameters: {other_params:,} ({100 * other_params / trainable_params:.2f}%)"
        )

    def save(model, model_dir, exp_name, suffix):
        """Helper function to save model states in a model-agnostic way"""
        save_path = os.path.join(model_dir, f"{exp_name}_{suffix}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")
