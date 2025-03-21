"""
Training and evaluation utilities for SpeechBCI multimodal language models
"""

import os
import torch
import gc
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from mmllm.data_utils import calculate_wer


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
    model_type="t5",
):
    """Evaluate model on a dataset, returning loss and both WER metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_original_texts = []  # NEW: collect original texts
    steps = 0

    # Set up language constraints based on model type
    generation_kwargs = {
        "max_length": max_gen_seq_len,
        "min_length": 2,
        "num_beams": num_gen_beams,
        "early_stopping": True,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "no_repeat_ngram_size": 2,
        "repetition_penalty": 1.2,
        "length_penalty": 1.0,
        "bad_words_ids": [[0]],
    }

    # Add model-specific parameters for English-only generation
    model_type = model_type.lower()
    if model_type == "mbart":
        # For multilingual BART: Force English BOS token
        generation_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["en_XX"]
    elif model_type == "t5":
        # For T5: Use decoder_start_token_id
        generation_kwargs["decoder_start_token_id"] = (
            model.t5_model.config.decoder_start_token_id
        )
    elif model_type == "bart":
        # For standard BART: Use the built-in BOS token
        generation_kwargs["decoder_start_token_id"] = tokenizer.bos_token_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Unpack batch
            inputs = batch["vqvae_embeddings"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_embeddings"].to(device)

            # Get original text from batch
            original_texts = batch["original_text"]

            # Forward pass
            outputs = model(
                inputs_embeds=inputs, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            steps += 1

            # Generate predictions with English constraints
            generated_ids = model.generate(
                inputs_embeds=inputs, attention_mask=attention_mask, **generation_kwargs
            )

            all_preds.extend(generated_ids.detach().cpu())
            all_original_texts.extend(original_texts)  # Use original text instead

            # Free memory
            del inputs, attention_mask, labels, outputs, generated_ids
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss = total_loss / steps
    std_wer, orig_wer = calculate_wer(
        all_preds, all_original_texts, tokenizer, model_dir, split_name
    )

    return avg_loss, std_wer, orig_wer


def save_model(model, model_dir, exp_name, suffix="best"):
    """Helper function to save model states in a model-agnostic way"""
    # Check model type and handle appropriately
    # First determine the model's core attribute (t5_model, bart_model, etc.)
    base_model_attr = None
    for attr in ["t5_model", "bart_model", "base_model"]:
        if hasattr(model, attr):
            base_model_attr = attr
            break
    
    # If we found a specific model attribute and it supports PEFT
    if base_model_attr and hasattr(getattr(model, base_model_attr), "save_pretrained"):
        # Create subfolder for PEFT model
        model_type = base_model_attr.split('_')[0]  # Extract model type (t5, bart)
        lora_path = os.path.join(model_dir, f"{exp_name}_{suffix}_lora")
        os.makedirs(lora_path, exist_ok=True)
        getattr(model, base_model_attr).save_pretrained(lora_path)
        
        # Save adapter separately
        adapter_path = os.path.join(model_dir, f"{exp_name}_{suffix}_adapter.pt")
        torch.save(model.input_adapter.state_dict(), adapter_path)
        
        print(f"Saved {model_type} model with PEFT/LoRA to {lora_path}")
    else:
        # Regular saving for non-PEFT models
        save_path = os.path.join(model_dir, f"{exp_name}_{suffix}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved regular model to {save_path}")


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
    model_type="t5",
):
    """Train and evaluate model, logging both standardized and original WER scores."""
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
        train_loss_eval, train_std_wer, train_orig_wer = evaluate(
            model,
            train_dl,
            tokenizer,
            device,
            max_gen_seq_len,
            num_gen_beams,
            model_dir,
            test_report_title,
            model_type=model_type,
        )
        writer.add_scalar("Loss/train_eval", train_loss_eval, epoch)
        writer.add_scalar("WER/train_standardized", train_std_wer, epoch)
        writer.add_scalar("WER/train_original", train_orig_wer, epoch)

        # Evaluation on test data
        test_report_title = f"{exp_name}_test_set_epoch_{epoch}"
        test_loss, test_std_wer, test_orig_wer = evaluate(
            model,
            test_dl,
            tokenizer,
            device,
            max_gen_seq_len,
            num_gen_beams,
            model_dir,
            test_report_title,
            model_type=model_type,
        )
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("WER/test_standardized", test_std_wer, epoch)
        writer.add_scalar("WER/test_original", test_orig_wer, epoch)

        print(
            f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, Train WER: {train_std_wer:.4f}/{train_orig_wer:.4f}, Test WER: {test_std_wer:.4f}/{test_orig_wer:.4f}"
        )

        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_model(model, model_dir, exp_name, suffix="best")

    # After training loop - save final model
    save_model(model, model_dir, exp_name, suffix="final")

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
