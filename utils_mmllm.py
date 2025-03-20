"""
Multimodal Language Model Utilities for Speech BCI

This module provides utilities for training and evaluating multimodal language models
that process Speech BCI data. It includes custom model adapters, training loops,
evaluation functions, and memory management utilities.

Key Components:
    - CustomEmbeddingT5: Adapter to connect VQVAE embeddings to T5 models
    - LoRA integration: Parameter-efficient fine-tuning
    - Training and evaluation loops with memory optimization
    - WER calculation and reporting utilities
"""

#
# Last updated: March 19, 2025
# Processing pipeline: etl.py → main_vqvae3D.py (training) → main_vqvae3D.py (encoding) → main_mmllm.py
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

# PEFT (Parameter-Efficient Fine-Tuning) imports
from peft import LoraConfig, get_peft_model, TaskType

# GPU memory management configuration
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_lora_model(base_model, r=16, alpha=32, dropout=0.1):
    """
    Apply LoRA configuration to a T5 model.

    Args:
        base_model: The base T5 model
        r: LoRA rank parameter (controls adaptation capacity)
        alpha: LoRA alpha scaling factor
        dropout: Dropout probability for LoRA layers

    Returns:
        PEFT model with LoRA applied to query and value matrices
    """
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,                    # Increase rank
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q", "v", "k", "o"],  # Add key and output matrices
        bias="lora_only",        # Add bias adaptation 
    )

    lora_model = get_peft_model(base_model, lora_config)
    return lora_model


class CustomEmbeddingT5(torch.nn.Module):
    """
    Adapter model that wraps T5 to accept custom embeddings as input.
    Works with both standard and LoRA-adapted T5 models.

    This adapter handles the dimensional mismatch between VQVAE embeddings
    and the expected input dimension of T5 models.
    """

    def __init__(self, t5_model, embedding_dim=64):
        """
        Initialize the adapter with a T5 model and projection layer.

        Args:
            t5_model: Base T5 model (standard or LoRA-adapted)
            embedding_dim: Dimension of input VQVAE embeddings
        """
        super().__init__()
        self.t5_model = t5_model
        self.input_adapter = torch.nn.Sequential(
torch.nn.Linear(embedding_dim, t5_model.config.d_model * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(t5_model.config.d_model * 2, t5_model.config.d_model)
)

    def forward(
        self, inputs_embeds, attention_mask, decoder_input_ids=None, labels=None
    ):
        """
        Forward pass through the adapter and T5 model.

        Args:
            inputs_embeds: VQVAE embeddings [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask for input sequence
            decoder_input_ids: Optional decoder input IDs for teacher forcing
            labels: Optional target labels for computing loss

        Returns:
            T5 model outputs
        """
        # Ensure inputs are fresh tensors
        inputs_embeds = inputs_embeds.detach()
        attention_mask = attention_mask.detach()
        if labels is not None:
            labels = labels.detach()

        # Map custom embeddings to T5 embedding dimension
        adapted_embeds = self.input_adapter(inputs_embeds)

        # Forward pass through T5 model with our adapted embeddings
        outputs = self.t5_model(
            inputs_embeds=adapted_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        return outputs

    def generate(self, inputs_embeds, attention_mask, **kwargs):
        """
        Generate text from input embeddings.

        Args:
            inputs_embeds: VQVAE embeddings [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask for input sequence
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs
        """
        adapted_embeds = self.input_adapter(inputs_embeds)
        return self.t5_model.generate(
            inputs_embeds=adapted_embeds, attention_mask=attention_mask, **kwargs
        )

    def print_trainable_parameters(self):
        """
        Print information about trainable parameters.

        Handles both standard T5 and PEFT models appropriately.
        """
        if hasattr(self.t5_model, "print_trainable_parameters"):
            return self.t5_model.print_trainable_parameters()
        else:
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            all_params = sum(p.numel() for p in self.parameters())
            print(f"Trainable parameters: {trainable_params}")
            print(f"Total parameters: {all_params}")
            print(f"Trainable%: {100 * trainable_params / all_params:.2f}%")

def calculate_wer(predictions, targets, tokenizer, model_dir, split_name):
    """
    Calculate Word Error Rate between predictions and targets.

    Args:
        predictions: Predicted token IDs
        targets: Target token IDs
        tokenizer: Tokenizer for decoding IDs to text
        model_dir: Directory to save prediction outputs
        split_name: Name of the data split (train/test/val)

    Returns:
        float: Mean WER score across all examples
    """
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_targets = tokenizer.batch_decode(targets, skip_special_tokens=True)

    # Save predictions and targets to file
    fname = os.path.join(model_dir, f"{split_name}_predictions.txt")
    print(f"calculate_wer {split_name} wrote file {fname}")
    with open(fname, "w") as f:
        for target, pred in zip(decoded_targets, decoded_preds):
            f.write(f"Target: {target}\n")
            f.write(f"Predicted: {pred}\n")
            f.write("\n")  # Add blank line between pairs

    # Calculate WER for each sentence pair
    wer_scores = [
        wer(target, pred) for target, pred in zip(decoded_targets, decoded_preds)
    ]
    return np.mean(wer_scores)


def run_exp(
    exp_name,
    train_dl,
    test_dl,
    val_dl,
    model,
    optimizer,
    tokenizer,
    device,
    num_epochs,
    max_gen_seq_len,
    num_gen_beams,
    model_dir,
    tensorboard_dir,
):
    """
    Train and evaluate a sequence-to-sequence model with memory optimizations.

    Args:
        exp_name: Experiment name for saving models and logs
        train_dl: Training data loader
        test_dl: Testing data loader
        val_dl: Validation data loader
        model: Model to train (CustomEmbeddingT5)
        optimizer: Optimizer for training
        tokenizer: Tokenizer for decoding predictions
        device: Device to run on (cuda/cpu)
        num_epochs: Number of training epochs
        max_gen_seq_len: Maximum sequence length for generation
        num_gen_beams: Number of beams for beam search generation
        model_dir: Directory to save model checkpoints
        tensorboard_dir: Directory for TensorBoard logs

    Returns:
        Trained model
    """
    writer = SummaryWriter(log_dir=tensorboard_dir)

    model = model.to(device)
    best_test_loss = float("inf")

    def reset_state():
        """Reset computation state to initial conditions"""
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # Reset any cached states in the model
        if hasattr(model, "t5_model"):
            for module in model.t5_model.modules():
                if hasattr(module, "cache_present"):
                    module.cache_present = False

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(num_epochs):
        # Reset state before epoch
        reset_state()

        # Training phase
        model.train()
        train_loss = train_epoch(model, train_dl, optimizer, device)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Reset state between phases
        reset_state()

        # Evaluation on training data (for diagnostics)
        model.eval()
        test_report_title = f"{exp_name}_training_set_epoch_{epoch}"
        train_loss_eval, train_wer = evaluate(
            model,
            train_dl,
            tokenizer,
            device,
            max_gen_seq_len,
            num_gen_beams,
            model_dir,
            test_report_title,
        )
        writer.add_scalar("Loss/train_eval", train_loss_eval, epoch)
        writer.add_scalar("WER/train", train_wer, epoch)

        # Evaluation on test data
        test_report_title = f"{exp_name}_test_set_epoch_{epoch}"
        test_loss, test_wer = evaluate(
            model,
            test_dl,
            tokenizer,
            device,
            max_gen_seq_len,
            num_gen_beams,
            model_dir,
            test_report_title,
        )
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("WER/test", test_wer, epoch)

        print(
            f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, Train WER: {train_wer:.4f}, Test WER: {test_wer:.4f}"
        )

        # Save the best model (special handling for LoRA models)
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
                torch.save(
                    model.state_dict(), os.path.join(model_dir, f"{exp_name}_best.pt")
                )

    # Save final model (special handling for LoRA models)
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

    # Final validation evaluation
    reset_state()
    test_report_title = f"{exp_name}_val_set_epoch_{epoch}"
    val_loss, val_wer = evaluate(
        model,
        val_dl,
        tokenizer,
        device,
        max_gen_seq_len,
        num_gen_beams,
        model_dir,
        test_report_title,
    )
    print(f"Validation Loss: {val_loss:.4f}, Validation WER: {val_wer:.4f}")
    writer.add_scalar("Loss/validation", val_loss, 0)
    writer.add_scalar("WER/validation", val_wer, 0)
    writer.close()
    return model


def train_epoch(model, dataloader, optimizer, device):
    """
    Standard training epoch without gradient accumulation.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer for parameter updates
        device: Device to run on (cuda/cpu)

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    steps = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Unpack batch
        inputs = batch["vqvae_embeddings"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label_embeddings"].to(device)

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(
            inputs_embeds=inputs, attention_mask=attention_mask, labels=labels
        )

        # Use full loss
        loss = outputs.loss

        # Backward pass with error handling
        try:
            loss.backward()
        except RuntimeError as e:
            if "backward through the graph a second time" in str(e):
                print("Warning: Graph reuse detected - using retain_graph")
                loss.backward(retain_graph=True)
            else:
                raise e

        # Optimize
        optimizer.step()

        # Track total loss
        total_loss += loss.item()
        steps += 1

        # Free memory
        del inputs, attention_mask, labels, outputs, loss
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = total_loss / steps
    return avg_loss


def evaluate(
    model,
    dataloader,
    tokenizer,
    device,
    max_gen_seq_len,
    num_gen_beams,
    model_dir,
    split_name,
):
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        tokenizer: Tokenizer for decoding predictions
        device: Device to run on (cuda/cpu)
        max_gen_seq_len: Maximum sequence length for generation
        num_gen_beams: Number of beams for beam search generation
        model_dir: Directory to save prediction outputs
        split_name: Name of data split for logging

    Returns:
        tuple: (average_loss, wer_score)
    """
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
                inputs_embeds=inputs, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            steps += 1

            # Generate predictions
            # MVP approach says hardcode values; later we optimize.
            generated_ids = model.generate(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                max_length=max_gen_seq_len,
                min_length=2,  # Prevent empty generations
                num_beams=num_gen_beams,
                early_stopping=True,
                do_sample=True,          # Enable sampling 
                temperature=0.7,         # Lower = more focused
                top_k=50,                # Consider top 50 tokens
                top_p=0.95,              # Nucleus sampling
                no_repeat_ngram_size=2,  # Prevent repetitive text
                repetition_penalty=1.2,  # Discourage repeating tokens
                length_penalty=1.0,  # Encourage longer outputs
                bad_words_ids=[[0]],  # Prevent generating only padding tokens
            )

            all_preds.extend(generated_ids.detach().cpu())
            all_targets.extend(labels.detach().cpu())

            # Free memory
            del inputs, attention_mask, labels, outputs, generated_ids
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss = total_loss / steps
    wer_score = calculate_wer(all_preds, all_targets, tokenizer, model_dir, split_name)

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
