"""
"""

#
# 14March2025 - actively working.
# Sequence: etl.py -> main_vqvae3D.py (training) -> main_vqvae3D.py (encoding) -> main_mmllm.py
#

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import gc
from transformers import T5ForConditionalGeneration, T5Config
from jiwer import wer
# Add PEFT imports
from peft import LoraConfig, get_peft_model, TaskType

# Add this to help with memory allocation
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_lora_model(base_model, r=8, alpha=32, dropout=0.1):
    """
    Apply LoRA configuration to a T5 model.
    
    Args:
        base_model: The base T5 model
        r: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout probability for LoRA layers
        
    Returns:
        PEFT model with LoRA applied
    """
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q", "v"],  # Apply to query and value matrices
        bias="none"
    )
    
    lora_model = get_peft_model(base_model, lora_config)
    return lora_model

class CustomEmbeddingT5(torch.nn.Module):
    """
    Adapter model that wraps T5 to accept custom embeddings as input.
    Works with both standard and LoRA-adapted T5 models.
    """
    def __init__(self, t5_model, embedding_dim=64):
        super().__init__()
        self.t5_model = t5_model
        self.input_adapter = torch.nn.Linear(embedding_dim, t5_model.config.d_model)
        
    def forward(self, inputs_embeds, attention_mask, decoder_input_ids=None, labels=None):
        # Map custom embeddings to T5 embedding dimension
        adapted_embeds = self.input_adapter(inputs_embeds)
        
        # Forward pass through T5 model with our adapted embeddings

        outputs = self.t5_model(
            inputs_embeds=adapted_embeds, 
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

        return outputs
    
    def generate(self, inputs_embeds, attention_mask, **kwargs):
        adapted_embeds = self.input_adapter(inputs_embeds)
        return self.t5_model.generate(
            inputs_embeds=adapted_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        
    def print_trainable_parameters(self):
        """Print information about trainable parameters."""
        if hasattr(self.t5_model, "print_trainable_parameters"):
            return self.t5_model.print_trainable_parameters()
        else:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.parameters())
            print(f"Trainable parameters: {trainable_params}")
            print(f"Total parameters: {all_params}")
            print(f"Trainable%: {100 * trainable_params / all_params:.2f}%")

def calculate_wer(predictions, targets, tokenizer):
    """Calculate Word Error Rate between predictions and targets"""
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_targets = tokenizer.batch_decode(targets, skip_special_tokens=True)
    
    # Calculate WER for each sentence pair
    wer_scores = [wer(target, pred) for target, pred in zip(decoded_targets, decoded_preds)]
    return np.mean(wer_scores)

# Modify run_exp to save LoRA weights
def run_exp(exp_name, train_dl, test_dl, val_dl, model, optimizer, tokenizer, device,
           num_epochs, model_dir, tensorboard_dir,
           gradient_accumulation_steps=4):
    """
    Train and evaluate a sequence-to-sequence model with memory optimizations.
    """
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    model = model.to(device)
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_dl, optimizer, device, 
                                gradient_accumulation_steps)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Clear cache between phases
        torch.cuda.empty_cache()
        gc.collect()
        
        # Testing - reduce batch size during evaluation if needed
        test_loss, test_wer = evaluate(model, test_dl, tokenizer, device, 
                                       batch_divider=2)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('WER/test', test_wer, epoch)
        
        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test WER: {test_wer:.4f}')
        
        # Save the best model - special handling for LoRA models
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            
            # Save LoRA weights if using PEFT
            if hasattr(model.t5_model, "save_pretrained"):
                # Create subfolder for PEFT model
                lora_path = os.path.join(model_dir, f"{exp_name}_best_lora")
                os.makedirs(lora_path, exist_ok=True)
                model.t5_model.save_pretrained(lora_path)
                
                # Save adapter separately for LinearLayer
                adapter_path = os.path.join(model_dir, f"{exp_name}_best_adapter.pt")
                torch.save(model.input_adapter.state_dict(), adapter_path)
            else:
                # Regular saving for non-PEFT models
                torch.save(model.state_dict(), os.path.join(model_dir, f"{exp_name}_best.pt"))
            
        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save final model - special handling for LoRA models
    if hasattr(model.t5_model, "save_pretrained"):
        # Create subfolder for PEFT model
        lora_path = os.path.join(model_dir, f"{exp_name}_final_lora")
        os.makedirs(lora_path, exist_ok=True)
        model.t5_model.save_pretrained(lora_path)
        
        # Save adapter separately
        adapter_path = os.path.join(model_dir, f"{exp_name}_final_adapter.pt")
        torch.save(model.input_adapter.state_dict(), adapter_path)
    else:
        # Regular saving for non-PEFT models
        torch.save(model.state_dict(), os.path.join(model_dir, f"{exp_name}_final.pt"))
    
    # Rest of function remains the same
    val_loss, val_wer = evaluate(model, val_dl, tokenizer, device, batch_divider=2)
    print(f'Validation Loss: {val_loss:.4f}, Validation WER: {val_wer:.4f}')
    writer.add_scalar('Loss/validation', val_loss, 0)
    writer.add_scalar('WER/validation', val_wer, 0)
    
    writer.close()
    return model

def train_epoch(model, dataloader, optimizer, device, 
               gradient_accumulation_steps=1):
    """Train for one epoch with memory optimizations"""
    model.train()
    total_loss = 0
    steps = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Unpack batch
        inputs, attention_mask, labels, labels_attention_mask = [b.to(device) for b in batch]
        
        # Forward pass
        outputs = model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            labels=labels
        )

        # Scale loss by accumulation steps
        loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
            
        # Track total loss
        total_loss += outputs.loss.item()
        steps += 1
        
        # Optimize every gradient_accumulation_steps or at the end of epoch
        if (i + 1) % gradient_accumulation_steps == 0 or i == len(dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            
        # Free up memory
        del inputs, attention_mask, labels, labels_attention_mask, outputs, loss
    
    avg_loss = total_loss / steps
    return avg_loss

def evaluate(model, dataloader, tokenizer, device, batch_divider=1):
    """Evaluate model with memory optimizations"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    steps = 0
    
    # Process in smaller sub-batches if needed
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Unpack batch
            inputs, attention_mask, labels, labels_attention_mask = [b.to(device) for b in batch]
            
            # Process in smaller chunks if batch_divider > 1
            if batch_divider > 1 and inputs.size(0) > 1:
                sub_batch_size = max(1, inputs.size(0) // batch_divider)
                for sub_idx in range(0, inputs.size(0), sub_batch_size):
                    end_idx = min(sub_idx + sub_batch_size, inputs.size(0))
                    sub_inputs = inputs[sub_idx:end_idx]
                    sub_attn = attention_mask[sub_idx:end_idx]
                    sub_labels = labels[sub_idx:end_idx]
                    
                    # Calculate loss
                    outputs = model(
                        inputs_embeds=sub_inputs,
                        attention_mask=sub_attn,
                        labels=sub_labels
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    steps += 1
                    
                    # Generate predictions with memory-efficient settings
                    generated_ids = model.generate(
                        inputs_embeds=sub_inputs,
                        attention_mask=sub_attn,
                        max_length=256,  # Reduced from 512
                        num_beams=2,     # Reduced from 4
                        early_stopping=True
                    )
                    
                    all_preds.extend(generated_ids.detach().cpu())
                    all_targets.extend(sub_labels.detach().cpu())
                    
                    # Free memory
                    del sub_inputs, sub_attn, sub_labels, outputs, generated_ids
                    torch.cuda.empty_cache()
            else:
                # Regular processing for smaller batches
                outputs = model(
                    inputs_embeds=inputs,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                steps += 1
                
                # Generate predictions with memory-efficient settings
                generated_ids = model.generate(
                    inputs_embeds=inputs,
                    attention_mask=attention_mask,
                    max_length=256,  # Reduced from 512
                    num_beams=2,     # Reduced from 4
                    early_stopping=True
                )
                
                all_preds.extend(generated_ids.detach().cpu())
                all_targets.extend(labels.detach().cpu())
            
            # Free memory
            del inputs, attention_mask, labels, labels_attention_mask
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / steps
    wer_score = calculate_wer(all_preds, all_targets, tokenizer)
    
    return avg_loss, wer_score

def get_vqvae_codebook_average(model):
    """
    Calculate the average of all embedding vectors in the VQ-VAE codebook.
    
    This function extracts the codebook embeddings from a trained VQ-VAE model
    and computes their mean. This can be useful for padding or initialization
    purposes when using the codebook representations.
    
    Args:
        model (VQVAE): A trained VQ-VAE model instance
        
    Returns:
        torch.Tensor: The average embedding vector with shape [embedding_dim]
    """
    # Get the vector quantizer (either standard or EMA)
    vq = model._vq_vae
    
    # Extract the embedding weights (codebook vectors)
    # Shape: [num_embeddings, embedding_dim]
    codebook = vq._embedding.weight.data
    
    # Calculate the average across all codebook vectors
    # Shape: [embedding_dim]
    avg_vector = torch.mean(codebook, dim=0)
    
    return avg_vector
