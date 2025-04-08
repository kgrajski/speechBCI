"""
Training and evaluation utilities for SpeechBCI multimodal language models
with support for both standard VQVAE embeddings and compressed representations.
"""

import os
import torch
import gc
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from mmllm.data_utils import calculate_wer
from transformers import BartForConditionalGeneration, BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Process generated text to ensure proper sentence format
def process_generated_output(text, model_type):
    """Process a single generated text.
    
    Args:
        text: Decoded text string
        model_type: Type of model (t5, bart, etc.)
        
    Returns:
        Processed text string
    """
    # For BART models, handle sentence tokens and ensure single sentence
    if model_type.lower() == "bart":
        # Remove sentence boundary tokens
        text = text.replace("<sentence>", "").replace("</sentence>", "").strip()
        
        # Ensure we have a single sentence with proper ending
        if "." in text:
            # Take first sentence and ensure it ends with period
            text = text.split(".")[0].strip() + "."
        elif len(text) > 0 and not text.endswith("."):
            # Add period if missing
            text = text.strip() + "."
    
    return text


def process_generated_texts(texts, model_type):
    """Process a list of generated texts based on model type.
    
    Args:
        texts: List of decoded text strings
        model_type: Type of model (t5, bart, etc.)
        
    Returns:
        List of processed texts
    """
    # Use list comprehension for cleaner, more functional approach
    return [process_generated_output(text, model_type) for text in texts]


def train_epoch(model, dataloader, optimizer, device, tokenizer, model_type, max_length=64, is_compressed=False):
    """
    Training epoch with support for both compressed and standard data formats.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer for parameter updates
        device: Device to run on (cuda/cpu)
        tokenizer: Tokenizer for encoding labels
        model_type: Type of model (t5, bart, etc.)
        max_length: Maximum sequence length for tokenization
        is_compressed: Whether using compressed data format

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    steps = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Zero gradients at start of step
        optimizer.zero_grad(set_to_none=True)
        
        # Handle different data formats
        if is_compressed:
            # Compressed data format
            inputs = batch["compressed_tokens"].to(device)
            
            # Get labels and apply special tokens for BART if needed
            raw_labels = batch["label"]
            if hasattr(model, "t5_model"):
                from transformers import T5Tokenizer
                tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
            else:
                from transformers import BartTokenizer
                tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            
            if model_type.lower() == "bart":
                # Wrap in sentence tokens
                labels = [f"<sentence>{label}</sentence>" for label in raw_labels]
            else:
                labels = raw_labels
            
            # Encode labels for the model
            encoded_labels = tokenizer(
                labels, 
                padding="longest",
                truncation=True,
                max_length=max_length,  # Use the parameter
                return_tensors="pt"
            ).to(device)
            
            # Forward pass with compressed inputs
            outputs = model(
                inputs_embeds=inputs,
                labels=encoded_labels.input_ids
            )
            
        else:
            # Standard VQVAE embedding format
            inputs = batch["vqvae_embeddings"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_embeddings"].to(device)

            # Forward pass with standard inputs
            outputs = model(
                inputs_embeds=inputs, 
                attention_mask=attention_mask, 
                labels=labels
            )

        # Get loss and update total
        loss = outputs.loss
        total_loss += loss.item()
        steps += 1

        # Backward pass with error handling
        try:
            loss.backward()
        except RuntimeError as e:
            if "backward through the graph a second time" in str(e):
                print("Warning: Graph reuse detected - using retain_graph")
                loss.backward(retain_graph=True)
            else:
                raise e

        # Apply gradients
        optimizer.step()

        # Free memory
        if is_compressed:
            del inputs, outputs, loss
            if 'encoded_labels' in locals():
                del encoded_labels
        else:
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
    is_compressed=False
):
    """
    Evaluate model on a dataset, supporting both compressed and standard data formats.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        tokenizer: Tokenizer for decoding predictions
        device: Device to run evaluation on
        max_gen_seq_len: Maximum sequence length for generation
        num_gen_beams: Number of beams for beam search
        model_dir: Directory to save evaluation results
        split_name: Name of the split being evaluated (e.g., "test", "val")
        model_type: Type of language model ("t5", "bart", etc.)
        is_compressed: Whether using compressed data format
        
    Returns:
        Tuple of (loss, standardized_wer, original_wer, unique_words)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_original_texts = [] 
    steps = 0

    # Set up language constraints based on model type
    generation_kwargs = {
        "max_length": max_gen_seq_len if max_gen_seq_len else 30,
        "min_length": 5,           # Encourage complete sentences
        "num_beams": num_gen_beams if num_gen_beams else 8,  
        "do_sample": False,        # Start with deterministic generation
        "length_penalty": 0.8,     # Favor shorter outputs
        "early_stopping": True
    }

    # Add model-specific parameters for generation
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
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}", leave=False):
            # Handle different data formats
            if is_compressed:
                # Compressed data format
                inputs = batch["compressed_tokens"].to(device)
                
                # Get original text and apply special tokens for BART if needed
                raw_texts = batch["label"]
                if model_type.lower() == "bart":
                    # Wrap in sentence tokens
                    texts_with_tokens = [f"<sentence>{text}</sentence>" for text in raw_texts]
                    # Use tokenizer-wrapped texts for loss calculation
                    encoded_labels = tokenizer(
                        texts_with_tokens if 'texts_with_tokens' in locals() else raw_texts, 
                        padding="longest", 
                        truncation=True, 
                        max_length=max_gen_seq_len,  # Use this parameter consistently
                        return_tensors="pt"
                    ).to(device)
                else:
                    encoded_labels = tokenizer(
                        raw_texts, 
                        padding="longest", 
                        truncation=True, 
                        max_length=max_gen_seq_len,  # Use this parameter consistently
                        return_tensors="pt"
                    ).to(device)
                
                # Keep original texts for WER calculation
                original_texts = raw_texts
                
                # Forward pass with compressed inputs
                outputs = model(
                    inputs_embeds=inputs,
                    labels=encoded_labels.input_ids
                )
                
                # Generate predictions
                generated_ids = model.generate(
                    inputs_embeds=inputs,
                    **generation_kwargs
                )
                
            else:
                # Standard VQVAE embedding format
                inputs = batch["vqvae_embeddings"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label_embeddings"].to(device)
                
                # Get original text
                original_texts = batch["original_text"]

                # Forward pass with standard inputs
                outputs = model(
                    inputs_embeds=inputs, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                # Generate predictions
                generated_ids = model.generate(
                    inputs_embeds=inputs, 
                    attention_mask=attention_mask, 
                    **generation_kwargs
                )

            # Extract loss
            loss = outputs.loss
            total_loss += loss.item()
            steps += 1

            # Collect predictions and original texts
            all_preds.extend(generated_ids.detach().cpu())
            all_original_texts.extend(original_texts)

            # Free memory
            if is_compressed:
                del inputs, outputs, generated_ids, encoded_labels
            else:
                del inputs, attention_mask, labels, outputs, generated_ids
                
            torch.cuda.empty_cache()
            gc.collect()

    # Calculate average loss
    avg_loss = total_loss / steps

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)

    # Post-process the decoded predictions
    processed_preds = process_generated_texts(decoded_preds, model_type)

    # Calculate WER metrics
    std_wer, orig_wer, num_uniq_pred_words = calculate_wer(
        processed_preds, 
        all_original_texts, 
        tokenizer, 
        model_dir, 
        split_name,
        already_decoded=True  # Predictions are already decoded
    )

    return avg_loss, std_wer, orig_wer, num_uniq_pred_words


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
    eval_training_set=False,
    is_compressed=False
):
    """
    Train and evaluate a multimodal language model with support for compressed data.
    
    Args:
        exp_name: Experiment name for logging
        train_dl: Training data loader
        test_dl: Testing data loader
        val_dl: Validation data loader
        model: Model to train
        optimizer: Optimizer for training
        tokenizer: Tokenizer for decoding predictions
        device: Device to run on (cuda/cpu)
        num_epochs: Number of training epochs
        max_gen_seq_len: Maximum sequence length for generation
        num_gen_beams: Number of beams for beam search
        model_dir: Directory to save models
        tensorboard_dir: Directory for TensorBoard logs
        model_type: Type of language model ("t5", "bart", etc.)
        eval_training_set: Whether to evaluate on training set
        is_compressed: Whether using compressed data format
        
    Returns:
        Trained model
    """
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Move model to device
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

    # Training loop
    for epoch in range(num_epochs):
        # Reset state before epoch
        reset_state()

        # Training phase
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(
            model, 
            train_dl, 
            optimizer, 
            device, 
            tokenizer,
            model_type,  # Add this parameter
            max_length=max_gen_seq_len,
            is_compressed=is_compressed
        )
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Reset state between phases
        reset_state()

        # Evaluation on training data (for diagnostics)
        if eval_training_set:
            model.eval()
            test_report_title = f"{exp_name}_training_set_epoch_{epoch}"
            train_loss_eval, train_std_wer, train_orig_wer, train_uniq_pred_words = evaluate(
                model,
                train_dl,
                tokenizer,
                device,
                max_gen_seq_len,
                num_gen_beams,
                model_dir,
                test_report_title,
                model_type=model_type,
                is_compressed=is_compressed
            )
            print(
                f"Training Set Evaluation - Loss: {train_loss_eval:.4f}, "
                f"WER: {train_std_wer:.4f}/{train_orig_wer:.4f}, "
                f"Unique words: {train_uniq_pred_words}"
            )
            writer.add_scalar("Loss/train_eval", train_loss_eval, epoch)
            writer.add_scalar("WER/train_standardized", train_std_wer, epoch)
            writer.add_scalar("WER/train_original", train_orig_wer, epoch)
            writer.add_scalar("Words/train_uniq_pred_words", train_uniq_pred_words, epoch)

        # Evaluation on test data
        test_report_title = f"{exp_name}_test_set_epoch_{epoch}"
        test_loss, test_std_wer, test_orig_wer, test_unique_pred_words = evaluate(
            model,
            test_dl,
            tokenizer,
            device,
            max_gen_seq_len,
            num_gen_beams,
            model_dir,
            test_report_title,
            model_type=model_type,
            is_compressed=is_compressed
        )
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("WER/test_standardized", test_std_wer, epoch)
        writer.add_scalar("WER/test_original", test_orig_wer, epoch)
        writer.add_scalar("Words/test_unique_pred_words", test_unique_pred_words, epoch)

        print(
            f"Test Set Evaluation - Loss: {test_loss:.4f}, "
            f"WER: {test_std_wer:.4f}/{test_orig_wer:.4f}, "
            f"Unique words: {test_unique_pred_words}"
        )

        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_model(model, model_dir, exp_name, suffix="best")
            print(f"New best model saved with test loss: {best_test_loss:.4f}")

    # After training loop - save final model
    save_model(model, model_dir, exp_name, suffix="final")
    print("Final model saved")

    # Final validation evaluation
    reset_state()
    test_report_title = f"{exp_name}_val_set_epoch_{num_epochs}"
    val_loss, val_std_wer, val_orig_wer, val_unique_pred_words = evaluate(
        model,
        val_dl,
        tokenizer,
        device,
        max_gen_seq_len,
        num_gen_beams,
        model_dir,
        test_report_title,
        model_type=model_type,
        is_compressed=is_compressed
    )
    print(
        f"Validation Set Evaluation - Loss: {val_loss:.4f}, "
        f"WER: {val_std_wer:.4f}/{val_orig_wer:.4f}, "
        f"Unique words: {val_unique_pred_words}"
    )
    writer.add_scalar("Loss/validation", val_loss, num_epochs)
    writer.add_scalar("WER/validation_standardized", val_std_wer, num_epochs)
    writer.add_scalar("WER/validation_original", val_orig_wer, num_epochs)
    writer.add_scalar("Words/val_unique_pred_words", val_unique_pred_words, num_epochs)
    
    # Close TensorBoard writer
    writer.close()
    return model
