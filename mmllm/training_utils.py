"""
Training and evaluation utilities for SpeechBCI multimodal language models
"""

import os
import torch
import gc
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from mmllm.data_utils import calculate_wer
from mmllm.data_utils import process_generated_texts, log_metrics


def training(
    description,
    mmllm,
    dataloader,
    optimizer,
    diversity_loss_weight,
    adapter_reg_weight,
    device,
):

    mmllm.to(device)
    mmllm.base_model.eval()  # Make sure model is in eval mode
    mmllm.input_adapter.train()  # Make sure adapter is in training mode

    total_loss = 0
    total_base_model_loss = 0
    total_diversity_loss = 0
    total_reg_loss = 0
    steps = 0

    for batch in tqdm(dataloader, desc=description):

        # Process batch - forward step
        # Unpack batch
        inputs = batch["vqvae_embeddings"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label_embeddings"].to(device)

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass through model
        adapter_outputs, mmllm_outputs = mmllm.forward(
            inputs,
            attention_mask,
            labels
        )
    
        # Compute losses
        diversity_loss = -torch.var(adapter_outputs, dim=1).mean()
        diversity_loss *= diversity_loss_weight
        reg_loss = torch.norm(adapter_outputs, p=2) / adapter_outputs.size(0)
        reg_loss *= adapter_reg_weight
        loss = mmllm_outputs.loss + diversity_loss + reg_loss

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Track losses
        total_loss += loss
        total_base_model_loss += mmllm_outputs.loss
        total_diversity_loss += diversity_loss
        total_reg_loss += reg_loss

        # Increment steps
        steps += 1
        
        # Clean up
        del inputs, attention_mask, labels, adapter_outputs, mmllm_outputs 

    # Calculate averages
    avg_total_loss = total_loss / steps
    avg_base_model_loss = total_base_model_loss / steps
    avg_diversity_loss = total_diversity_loss / steps
    avg_reg_loss = total_reg_loss / steps
    
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()

    # Return  losses
    losses = {
        "total": avg_total_loss,
        "main": avg_base_model_loss,
        "diversity": avg_diversity_loss,
        "reg": avg_reg_loss,
    }
    return losses

def generation(
    description,
    mmllm,
    dataloader,
    model_dir,
    device,
):
    mmllm.to(device)
    mmllm.base_model.eval()  # Make sure model is in eval mode
    mmllm.input_adapter.eval()  # Make sure adapter is in training mode

    all_preds = []
    all_original_texts = []

    # Set up language constraints based on model type
    # Let's over-specify the generation_kwargs for now
    task_prompt = "Generate a sentence: "  # Define the task prompt here
    generation_kwargs = {
        "max_length": 64,
        "min_length": 5,
        "num_beams": 4,
        "do_sample": False,
        "length_penalty": 0.8,
        "early_stopping": True,
        "task_prompt": task_prompt,  # Pass the task prompt
        "num_return_sequences": 1,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "tokenizer": mmllm.tokenizer,  # Pass tokenizer to generation
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description):
            # Unpack batch

            inputs = batch["vqvae_embeddings"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_embeddings"].to(device)
            original_texts = batch["original_text"]

            # Generate predictions using the mmllm.generate method
            generated_ids = mmllm.generate(
                input_embeddings=inputs,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

            all_preds.extend(generated_ids.detach().cpu().tolist())
            all_original_texts.extend(original_texts)

            # Free memory
            del inputs, attention_mask, labels, generated_ids

            torch.cuda.empty_cache()
            gc.collect()

    # Decode and process the generated texts
    decoded_preds = mmllm.tokenizer.batch_decode(all_preds, skip_special_tokens=True)
    processed_preds = process_generated_texts(
        decoded_preds, mmllm.model_type, task_prompt
    )  # Pass the task prompt here

    # Calculate WER
    std_wer, orig_wer, num_uniq_pred_words = calculate_wer(
        processed_preds,
        all_original_texts,
        mmllm.tokenizer,
        model_dir,
        description,
        already_decoded=True,
    )

    # Clean up
    torch.cuda.empty_cache()
    gc.collect()

    # Return the results
    return {
        "std_wer": std_wer,
        "orig_wer": orig_wer,
        "num_uniq_pred_words": num_uniq_pred_words,
    }
    

def run_exp(
    exp_name,
    mmllm,  # model is an instance of MultimodalLLM: adapter + base
    device,
    train_dl,
    test_dl,
    val_dl,
    optimizer,
    num_epochs,
    diversity_loss_weight,
    adapter_reg_weight,
    model_dir,
    tensorboard_dir,
):

    # Push the adapter and model to the device (probabaly redundant, but no harm)
    mmllm = mmllm.to(device)

    # Set reference test_loss used to decide whether and when to write a model
    best_gen_test = float("inf")

    # Start TensorBoard writer and start by logging hyperparameters``
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Clean-up before starting
    torch.cuda.empty_cache()
    gc.collect()

    # Conduct the main loop
    for epoch in range(num_epochs):

        # Evaluate losses on training data and do a training step
        description = "loss_train"
        train_loss = training(
            description,
            mmllm,
            train_dl,
            optimizer,
            diversity_loss_weight,
            adapter_reg_weight,
            device,
        )
        log_metrics(writer, exp_name, description, train_loss, epoch)

        # Generate and log word-level performance on the test set
        description = "gen_test"
        gen_test = generation(
            description,
            mmllm,
            test_dl,
            model_dir,
            device,
        )
        log_metrics(writer, exp_name, description, gen_test, epoch)

        # Save the best model
        if gen_test["std_wer"] < best_gen_test:
            best_gen_test = gen_test["std_wer"]
            mmllm.save(model_dir, exp_name, suffix="best_gen_test")

    # After training loop - save final model
    mmllm.save(model_dir, exp_name, suffix="final")

        # Generate and log word-level performance on the validation set
    description = "gen_val"
    gen_val = generation(
        description,
        mmllm,
        val_dl,
        model_dir,
        device,
    )
    log_metrics(writer, exp_name, description, gen_val, epoch)

    # Close the TensorBoard writer
    writer.flush()
    writer.close()

    # Clean-up before exiting
    torch.cuda.empty_cache()
    gc.collect()

    # Return the model
    return mmllm
