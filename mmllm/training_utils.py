"""
Training and evaluation utilities for SpeechBCI multimodal language models
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


# Modify train_epoch to include additional losses
def train_epoch(model, adapter, dataloader, optimizer, device, diversity_loss_weight=0.0, adapter_reg_weight=0.0, writer=None):
    model.train()
    adapter.train()  # Make sure adapter is in training mode
    
    total_loss = 0
    total_main_loss = 0
    total_diversity_loss = 0
    total_reg_loss = 0
    steps = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Unpack batch
        inputs = batch["vqvae_embeddings"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label_embeddings"].to(device)

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Process inputs through adapter
        adapter_outputs = adapter(inputs)
        
        # Forward pass through model
        outputs = model(
            inputs_embeds=adapter_outputs,
            attention_mask=attention_mask,
            labels=labels
        )

        # Main language modeling loss
        main_loss = outputs.loss
        
        # Initialize additional losses
        diversity_loss = 0.0
        reg_loss = 0.0

        # Calculate diversity loss if enabled
        if diversity_loss_weight > 0:
            # Calculate variance across feature dimension (higher variance = more diversity)
            diversity_loss = -torch.var(adapter_outputs, dim=1).mean()
            
        # Calculate regularization loss if enabled
        if adapter_reg_weight > 0:
            # L2 regularization on the adapter output
            reg_loss = torch.norm(adapter_outputs, p=2) / adapter_outputs.size(0)

        # Combine losses
        loss = main_loss + diversity_loss_weight * diversity_loss + adapter_reg_weight * reg_loss

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        
        if diversity_loss_weight > 0:
            total_diversity_loss += diversity_loss.item()
        
        if adapter_reg_weight > 0:
            total_reg_loss += reg_loss.item()
            
        steps += 1

        # Log component losses every 20 steps
        if writer is not None and steps % 20 == 0:
            writer.add_scalar("Loss/step/main", main_loss.item(), steps)
            if diversity_loss_weight > 0:
                writer.add_scalar("Loss/step/diversity", diversity_loss.item(), steps)
            if adapter_reg_weight > 0:
                writer.add_scalar("Loss/step/reg", reg_loss.item(), steps)
            writer.add_scalar("Loss/step/total", loss.item(), steps)

        # Free memory
        del inputs, attention_mask, labels, outputs, loss
        if diversity_loss_weight > 0:
            del diversity_loss
        if adapter_reg_weight > 0:
            del reg_loss
        torch.cuda.empty_cache()
        gc.collect()

    # Calculate averages
    avg_loss = total_loss / steps
    avg_main_loss = total_main_loss / steps
    avg_diversity_loss = total_diversity_loss / steps if diversity_loss_weight > 0 else 0
    avg_reg_loss = total_reg_loss / steps if adapter_reg_weight > 0 else 0
    
    # Return all losses
    return {
        'total': avg_loss,
        'main': avg_main_loss,
        'diversity': avg_diversity_loss,
        'reg': avg_reg_loss
    }


def evaluate(
    model,
    adapter,  # Added adapter parameter
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
    adapter.eval()  # Set adapter to eval mode
    
    total_loss = 0
    all_preds = []
    all_original_texts = []
    steps = 0

    # Set up language constraints based on model type
    generation_kwargs = {
        "max_length": 30,
        "min_length": 5,
        "num_beams": 8,
        "do_sample": False,
        "length_penalty": 0.8,
        "early_stopping": True
    }

    # Model-specific parameters unchanged...
    model_type = model_type.lower()
    if model_type == "mbart":
        generation_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["en_XX"]
    elif model_type == "t5":
        generation_kwargs["decoder_start_token_id"] = (
            model.t5_model.config.decoder_start_token_id
        )
    elif model_type == "bart":
        generation_kwargs["decoder_start_token_id"] = tokenizer.bos_token_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Unpack batch
            inputs = batch["vqvae_embeddings"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_embeddings"].to(device)
            original_texts = batch["original_text"]

            # Process inputs through adapter first
            adapter_outputs = adapter(inputs)
            
            # Forward pass using adapter outputs
            outputs = model(
                inputs_embeds=adapter_outputs,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            steps += 1

            # Generate predictions using adapter outputs
            generated_ids = model.generate(
                inputs_embeds=adapter_outputs,  # Use adapter outputs for generation
                attention_mask=attention_mask,
                **generation_kwargs
            )

            all_preds.extend(generated_ids.detach().cpu())
            all_original_texts.extend(original_texts)

            # Free memory
            del inputs, attention_mask, labels, outputs, generated_ids, adapter_outputs
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss = total_loss / steps

    # Rest of the function unchanged...
    decoded_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
    processed_preds = process_generated_texts(decoded_preds, model_type)
    std_wer, orig_wer, num_uniq_pred_words = calculate_wer(
        processed_preds, 
        all_original_texts, 
        tokenizer, 
        model_dir, 
        split_name,
        already_decoded=True
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
        
        # Save adapter separately if it exists
        if hasattr(model, "input_adapter"):
            adapter_path = os.path.join(model_dir, f"{exp_name}_{suffix}_adapter.pt")
            torch.save(model.input_adapter.state_dict(), adapter_path)
            print(f"Saved adapter to {adapter_path}")
        
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
    adapter,  # Added adapter parameter
    optimizer,
    tokenizer,
    device,
    num_epochs,
    max_gen_seq_len,
    num_gen_beams,
    model_dir,
    tensorboard_dir,
    model_type,
    eval_training_set,
    diversity_loss_weight,
    adapter_reg_weight,
    diagnostics,
):
    """Train and evaluate model with additional loss components and optional diagnostics."""
    writer = SummaryWriter(log_dir=tensorboard_dir)

    model = model.to(device)
    adapter = adapter.to(device)  # Ensure adapter is on the correct device
    best_test_loss = float("inf")
    
    # Log hyperparameters
    writer.add_text("Hyperparameters/diversity_loss_weight", str(diversity_loss_weight), 0)
    writer.add_text("Hyperparameters/adapter_reg_weight", str(adapter_reg_weight), 0)

    # Initialize diagnostic step counter
    diagnostic_steps = 0

    for epoch in range(num_epochs):
        # Training phase with additional loss components
        model.train()
        adapter.train()
        train_losses = train_epoch(
            model, 
            adapter,  # Pass adapter
            train_dl, 
            optimizer, 
            device,
            diversity_loss_weight=diversity_loss_weight,
            adapter_reg_weight=adapter_reg_weight,
            writer=writer
        )
        
        # Log all loss components
        writer.add_scalar("Loss/train/total", train_losses['total'], epoch)
        writer.add_scalar("Loss/train/main", train_losses['main'], epoch)
        if diversity_loss_weight > 0:
            writer.add_scalar("Loss/train/diversity", train_losses['diversity'], epoch)
        if adapter_reg_weight > 0:
            writer.add_scalar("Loss/train/reg", train_losses['reg'], epoch)

        # Run diagnostics if enabled (every 2 epochs to save time)
        if diagnostics is not None and epoch % 2 == 0:
            diagnostics.capture_epoch_diagnostics(model, train_dl, epoch)
            
        # Evaluation on training data (for diagnostics)
        if eval_training_set:
            model.eval()
            adapter.eval()
            test_report_title = f"{exp_name}_training_set_epoch_{epoch}"
            train_loss_eval, train_std_wer, train_orig_wer, train_uniq_pred_words = evaluate(
                model,
                adapter,  # Add adapter parameter
                train_dl, 
                tokenizer, 
                device, 
                max_gen_seq_len, 
                num_gen_beams, 
                model_dir, 
                test_report_title, 
                model_type=model_type,
            )
            print(
                f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss_eval:.4f}, "
                f"Train WER: {train_std_wer:.4f}/{train_orig_wer:.4f}",
                f"Unique pred words: {train_uniq_pred_words}",
            )
            writer.add_scalar("Loss/train_eval", train_loss_eval, epoch)
            writer.add_scalar("WER/train_standardized", train_std_wer, epoch)
            writer.add_scalar("WER/train_original", train_orig_wer, epoch)
            writer.add_scalar("Words/train_uniq_pred_words", train_uniq_pred_words, epoch)

        # Evaluation on test data
        test_report_title = f"{exp_name}_test_set_epoch_{epoch}"
        test_loss, test_std_wer, test_orig_wer, test_unique_pred_words = evaluate(
            model,
            adapter,  # Add adapter parameter
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
        writer.add_scalar("Words/test_unique_pred_words", test_unique_pred_words, epoch)

        # Print detailed loss breakdown
        loss_details = f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_losses['total']:.4f} "
        loss_details += f"(Main: {train_losses['main']:.4f}"
        if diversity_loss_weight > 0:
            loss_details += f", Div: {train_losses['diversity']:.4f}"
        if adapter_reg_weight > 0:
            loss_details += f", Reg: {train_losses['reg']:.4f}"
        loss_details += f"), Test Loss: {test_loss:.4f}"
        print(loss_details)

        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_model(model, model_dir, exp_name, suffix="best")

    # After training loop - save final model
    save_model(model, model_dir, exp_name, suffix="final")

    # Final validation evaluation
    model.eval()
    adapter.eval()
    with torch.no_grad():
        for batch in val_dl:
            inputs = batch["vqvae_embeddings"].to(device)
            attention_mask = batch["attention_mask"].to(device) 
            labels = batch["label_embeddings"].to(device)
            
            # Apply adapter
            adapter_outputs = adapter(inputs)
            
            # Pass to model
            outputs = model(
                inputs_embeds=adapter_outputs,
                attention_mask=attention_mask,
                labels=labels
            )
            # Rest of evaluation code...

    test_report_title = f"{exp_name}_val_set_epoch_{epoch}"
    val_loss, val_std_wer, val_orig_wer, val_unique_pred_words = evaluate(
        model,
        adapter,  # Add adapter parameter
        val_dl, 
        tokenizer, 
        device, 
        max_gen_seq_len, 
        num_gen_beams, 
        model_dir, 
        test_report_title, 
        model_type=model_type,
    )
    print(
        f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {val_loss:.4f}, "
        f"Train WER: {val_std_wer:.4f}/{val_orig_wer:.4f}",
        f"Unique pred words: {val_unique_pred_words}",
    )
    writer.add_scalar("Loss/validation", val_loss, epoch)
    writer.add_scalar("WER/validation_standardized", val_std_wer, epoch)
    writer.add_scalar("WER/validation_original", val_orig_wer, epoch)
    writer.add_scalar("Words/val_unique_pred_words", val_unique_pred_words, epoch)
    
    if diagnostics is not None:
        print("\n==== Running Final Diagnostics ====")
        diagnostics.run_all_diagnostics(test_dl)
        diagnostics.visualize_attention_patterns()
        diagnostics.analyze_mode_collapse()
        
    writer.close()
    return model
