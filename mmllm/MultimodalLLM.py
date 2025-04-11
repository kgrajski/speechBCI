import os
import torch
import torch.nn as nn


class MultimodalLLM(nn.Module):
    """
    Base class for multimodal language models with various adapters.
    Note the design principle of clean separatetion of model and adapter.
    Operates on one input batch at a time.
    """

    def __init__(self, adapter_type, model_type, input_adapter, base_model, tokenizer):
        super().__init__()
        self.adapter_type = adapter_type 
        self.model_type = model_type
        self.input_adapter = input_adapter
        self.base_model = base_model
        self.tokenizer = tokenizer

    def forward(
        self,
        input_embeddings,
        attention_mask,
        labels,
        **kwargs,
    ):

        # Forward through the adapter
        adapter_outputs = self.input_adapter(input_embeddings)

        # Forward through base model
        mmllm_outputs = self.base_model(
            inputs_embeds=adapter_outputs,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return adapter_outputs, mmllm_outputs

    def generate(
        self,
        input_embeddings,
        attention_mask,
        **kwargs,
    ):

        # Forward through adapter
        adapter_outputs = self.input_adapter(input_embeddings)

        # Add task-specific prompt if provided
        tokenizer = kwargs.pop("tokenizer", None)
        prompt_tokens = tokenizer(
            kwargs.pop("task_prompt", None), 
            return_tensors="pt").input_ids.to(
            adapter_outputs.device
        )
        prompt_embeds = self.base_model.get_input_embeddings()(prompt_tokens)

        # Repeat prompt for each item in batch
        batch_size = adapter_outputs.shape[0]
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)

        # Concatenate with adapter outputs
        combined_embeds = torch.cat([prompt_embeds, adapter_outputs], dim=1)

        # Update attention mask if provided
        if attention_mask is not None:
            prompt_attention = torch.ones(
                batch_size, prompt_tokens.shape[1], device=attention_mask.device
            ).to(attention_mask.device)
            attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

        # Generate with processed embeddings
        return self.base_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
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
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Updated breakdown - no adapter inside MMLLM
        adapter_params = sum(p.numel() for n, p in self.named_parameters() 
                        if 'input_adapter' in n and p.requires_grad)
        other_params = trainable_params - adapter_params
        
        print(f"  - Adapter parameters: {adapter_params:,} ({100 * adapter_params / trainable_params:.2f}%)")
        print(f"  - Other parameters: {other_params:,} ({100 * other_params / trainable_params:.2f}%)")

    def save(model, model_dir, exp_name, suffix):
        """Helper function to save model states in a model-agnostic way"""
        save_path = os.path.join(model_dir, f"{exp_name}_{suffix}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")
