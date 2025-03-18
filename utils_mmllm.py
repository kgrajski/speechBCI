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

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from transformers import T5ForConditionalGeneration, T5Config
from jiwer import wer

class CustomEmbeddingT5(torch.nn.Module):
    """
    Adapter model that wraps T5 to accept custom embeddings as input.
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

def calculate_wer(predictions, targets, tokenizer):
    """Calculate Word Error Rate between predictions and targets"""
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_targets = tokenizer.batch_decode(targets, skip_special_tokens=True)
    
    # Calculate WER for each sentence pair
    wer_scores = [wer(target, pred) for target, pred in zip(decoded_targets, decoded_preds)]
    return np.mean(wer_scores)

def run_exp(exp_name, train_dl, test_dl, val_dl, model, optimizer, tokenizer, device,
           num_epochs, learning_rate, model_dir, tensorboard_dir):
    """
    Train and evaluate a sequence-to-sequence model.
    
    Args:
        exp_name (str): Experiment name
        train_dl (DataLoader): Training data loader
        test_dl (DataLoader): Testing data loader
        val_dl (DataLoader): Validation data loader
        model (CustomEmbeddingT5): Model to train
        optimizer (torch.optim.Optimizer): Optimizer
        tokenizer: T5 tokenizer for decoding predictions
        device (torch.device): Device to run on
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate
        model_dir (str): Directory to save models
        tensorboard_dir (str): Directory for tensorboard logs
    """
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    model = model.to(device)
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_dl, optimizer, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Testing
        test_loss, test_wer = evaluate(model, test_dl, tokenizer, device)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('WER/test', test_wer, epoch)
        
        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test WER: {test_wer:.4f}')
        
        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(model_dir, f"{exp_name}_best.pt"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, f"{exp_name}_final.pt"))
    
    # Evaluate on validation set
    val_loss, val_wer = evaluate(model, val_dl, tokenizer, device)
    print(f'Validation Loss: {val_loss:.4f}, Validation WER: {val_wer:.4f}')
    writer.add_scalar('Loss/validation', val_loss, 0)
    writer.add_scalar('WER/validation', val_wer, 0)
    
    writer.close()
    return model

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Unpack batch
        inputs, attention_mask, labels, labels_attention_mask = [b.to(device) for b in batch]
        
        # Forward pass
        outputs = model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, tokenizer, device):
    """Evaluate model and calculate metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Unpack batch
            inputs, attention_mask, labels, labels_attention_mask = [b.to(device) for b in batch]
            
            # Calculate loss
            outputs = model(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Generate predictions
            generated_ids = model.generate(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            
            all_preds.extend(generated_ids.detach().cpu())
            all_targets.extend(labels.detach().cpu())
    
    avg_loss = total_loss / len(dataloader)
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
