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
    """Trains the multimodal language model.

    Args:
        description (str): Description of the training run.
        mmllm (MultimodalLLM): The multimodal language model to train.
        dataloader (DataLoader): The data loader for the training data.
        optimizer (Optimizer): The optimizer to use for training.
        diversity_loss_weight (float): Weight for the diversity loss. Encourages diverse
            representations to avoid mode collapse.
        adapter_reg_weight (float): Weight for the adapter regularization loss. Prevents
            overfitting by penalizing large adapter weights.
        device (torch.device): The device to train on (CPU or GPU).

    Returns:
        dict: A dictionary containing the average training losses.
            "total": Average total loss.
            "main": Average base model loss (cross-entropy).
            "diversity": Average diversity loss.
            "reg": Average adapter regularization loss.
    """

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
        decoded_preds,
        mmllm.model_type,
        task_prompt
    )

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
    num_epochs,
    learning_rate,
    diversity_loss_weight,
    adapter_reg_weight,
    model_dir,
    tensorboard_dir,
    momentum=0.9,
    lr_decay_factor=0.5,
    lr_decay_epochs=10,
    loss_plateau_epochs=5,
    loss_threshold=0.001,
):
    """Runs the multimodal language model experiment.

    Args:
        exp_name (str): The name of the experiment.
        mmllm (MultimodalLLM): The multimodal language model to train.
        device (torch.device): The device to train on (CPU or GPU).
        train_dl (DataLoader): The data loader for the training data.
        test_dl (DataLoader): The data loader for the test data.
        val_dl (DataLoader): The data loader for the validation data.
        num_epochs (int): The number of epochs to train for.
        learning_rate (float): The initial learning rate.
        diversity_loss_weight (float): Weight for the diversity loss.
        adapter_reg_weight (float): Weight for the adapter regularization loss.
        model_dir (str): The directory to save the model checkpoints.
        tensorboard_dir (str): The directory to save the TensorBoard logs.
        momentum (float): Momentum factor for the Adam optimizer.
        lr_decay_factor (float): Factor by which to decay the learning rate.
        lr_decay_epochs (int): Number of epochs after which to decay the learning rate.
        loss_plateau_epochs (int): Number of epochs to wait for loss improvement.
        loss_threshold (float): Minimum loss improvement to consider.

    Returns:
        MultimodalLLM: The trained multimodal language model.
    """

    # Push the adapter and model to the device (probabaly redundant, but no harm)
    mmllm = mmllm.to(device)

    # Set reference test_loss used to decide whether and when to write a model
    best_gen_test = float("inf")

    # Start TensorBoard writer and start by logging hyperparameters``
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Clean-up before starting
    torch.cuda.empty_cache()
    gc.collect()
    
        # Set up optimizer with parameters from the adapter - that is all we're training
    optimizer_params = list(mmllm.input_adapter.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=learning_rate,
        betas=(momentum, 0.999))

    # Learning rate scheduler setup
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_epochs,
        gamma=lr_decay_factor)

    # Track loss history
    best_train_loss = float('inf')
    epochs_since_improvement = 0

    # Conduct the main loop
    for epoch in range(num_epochs):

        # Evaluate losses on training data and do a training step
        description = "loss_train" + f"_{epoch}"
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

        # Check for loss plateau
        log_metrics(writer, exp_name, 'learing_rate', optimizer.param_groups, epoch)
        if train_loss["total"] < best_train_loss - loss_threshold:
            best_train_loss = train_loss["total"]
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= loss_plateau_epochs:
                # Reduce learning rate if loss plateaus
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay_factor
                print(f"Epoch {epoch}: Loss plateau: lr to {param_group['lr']}")
                epochs_since_improvement = 0  # Reset counter

        # Generate and log word-level performance on the test set
        description = "gen_test" + f"_{epoch}"
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

        # Step the scheduler
        scheduler.step()

    # After training loop - save final model
    mmllm.save(model_dir, exp_name, suffix="final")

        # Generate and log word-level performance on the validation set
    description = "gen_val" + f"_{epoch}"
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
