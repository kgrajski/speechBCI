import gc
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartConfig
from jiwer import wer
# Add PEFT imports
from peft import LoraConfig, get_peft_model, TaskType

# Add this to help with memory allocation
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def get_lora_model(base_model, r=8, alpha=32, dropout=0.1):
    """Apply LoRA adaptation to a BART model"""
    # Configure LoRA for encoder and decoder
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,                          # LoRA attention dimension
        lora_alpha=alpha,             # LoRA scaling factor
        lora_dropout=dropout,         # Dropout probability for LoRA layers
        target_modules=["q_proj", "v_proj"],  # Target specific modules in BART
        bias="none",
        inference_mode=False,         # Enable training
    )
    
    lora_model = get_peft_model(base_model, lora_config)
    return lora_model

class CustomEmbeddingBART(torch.nn.Module):
    """
    Adapter model that wraps BART to accept custom embeddings as input.
    Works with both standard and LoRA-adapted BART models.
    """
    def __init__(self, bart_model, embedding_dim=64):
        super().__init__()
        self.bart_model = bart_model
        self.input_adapter = torch.nn.Linear(embedding_dim, bart_model.config.d_model)
        
    def forward(self, inputs_embeds, attention_mask, decoder_input_ids=None, labels=None):
        # Map custom embeddings to BART embedding dimension
        adapted_embeds = self.input_adapter(inputs_embeds)
        
        # Forward pass through BART model with our adapted embeddings
        outputs = self.bart_model(
            inputs_embeds=adapted_embeds, 
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

        return outputs
    
    def generate(self, inputs_embeds, attention_mask, **kwargs):
        adapted_embeds = self.input_adapter(inputs_embeds)
        return self.bart_model.generate(
            inputs_embeds=adapted_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        
    def print_trainable_parameters(self):
        """Print information about trainable parameters."""
        if hasattr(self.bart_model, "print_trainable_parameters"):
            return self.bart_model.print_trainable_parameters()
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

def run_exp(exp_name, train_dl, test_dl, val_dl, model, optimizer, tokenizer, device,
            num_epochs, max_gen_seq_len, num_gen_beams, model_dir, tensorboard_dir):
    """
    Train and evaluate a sequence-to-sequence model with memory optimizations.
    """
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    model = model.to(device)
    best_test_loss = float('inf')
    
    def reset_state():
        """Reset computation state to initial conditions"""
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(num_epochs):
        # Reset state before epoch
        reset_state()
        
        # Training
        model.train()
        train_loss = train_epoch(model, train_dl, optimizer, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Reset state between phases
        reset_state()
        
        # Testing
        model.eval()
        test_loss, test_wer = evaluate(model, test_dl, tokenizer, device, max_gen_seq_len, num_gen_beams)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('WER/test', test_wer, epoch)
        
        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test WER: {test_wer:.4f}')
        
        # Save model if it's the best so far
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            
            # Save the model
            if hasattr(model.bart_model, "save_pretrained"):
                # Create subfolder for PEFT model
                lora_path = os.path.join(model_dir, f"{exp_name}_best_lora")
                os.makedirs(lora_path, exist_ok=True)
                model.bart_model.save_pretrained(lora_path)
                
                # Save adapter separately
                adapter_path = os.path.join(model_dir, f"{exp_name}_best_adapter.pt")
                torch.save(model.input_adapter.state_dict(), adapter_path)
            else:
                # Regular saving for non-PEFT models
                torch.save(model.state_dict(), os.path.join(model_dir, f"{exp_name}_best.pt"))
            
        # Reset state after epoch
        reset_state()
        
    # Save final model - special handling for LoRA models
    if hasattr(model.bart_model, "save_pretrained"):
        # Create subfolder for PEFT model
        lora_path = os.path.join(model_dir, f"{exp_name}_final_lora")
        os.makedirs(lora_path, exist_ok=True)
        model.bart_model.save_pretrained(lora_path)
        
        # Save adapter separately
        adapter_path = os.path.join(model_dir, f"{exp_name}_final_adapter.pt")
        torch.save(model.input_adapter.state_dict(), adapter_path)
    else:
        # Regular saving for non-PEFT models
        torch.save(model.state_dict(), os.path.join(model_dir, f"{exp_name}_final.pt"))
    
    reset_state()
    val_loss, val_wer = evaluate(model, val_dl, tokenizer, device, max_gen_seq_len, num_gen_beams)
    print(f'Validation Loss: {val_loss:.4f}, Validation WER: {val_wer:.4f}')
    writer.add_scalar('Loss/validation', val_loss, 0)
    writer.add_scalar('WER/validation', val_wer, 0)
    
    writer.close()
    return model

def train_epoch(model, dataloader, optimizer, device):
    """Standard training epoch with improved error handling"""
    model.train()
    total_loss = 0
    steps = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad(set_to_none=True)
        
        # Unpack batch
        inputs = batch["vqvae_embeddings"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label_embeddings"].to(device)
        
        # Forward pass
        outputs = model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass with error handling
        try:
            loss.backward()
        except RuntimeError as e:
            if "Trying to backward through the graph a second time" in str(e):
                print("Warning: Graph reuse detected - using retain_graph=True")
                loss.backward(retain_graph=True)
            else:
                raise e
                
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimize
        optimizer.step()
        
        # Track total loss
        total_loss += loss.item()
        steps += 1
        
        # Free memory
        del inputs, attention_mask, labels, outputs, loss
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / steps
    return avg_loss

def evaluate(model, dataloader, tokenizer, device, max_gen_seq_len=32, num_gen_beams=5):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Unpack batch
            inputs = batch["vqvae_embeddings"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_embeddings"].to(device)
            
            # Forward pass
            outputs = model(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            steps += 1
            
            # Generate predictions
            generated_ids = model.generate(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                max_length=max_gen_seq_len,
                num_beams=num_gen_beams,
                early_stopping=True
            )
            
            all_preds.extend(generated_ids)
            all_targets.extend(labels)
            
            # Free memory
            del inputs, attention_mask, labels, outputs, loss, generated_ids
            torch.cuda.empty_cache()
    
    # Calculate WER
    wer_score = calculate_wer(all_preds, all_targets, tokenizer)
    avg_loss = total_loss / steps if steps > 0 else float('inf')
    
    return avg_loss, wer_score