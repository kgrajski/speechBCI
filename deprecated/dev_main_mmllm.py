"""
Multimodal Language Model with Hierarchically Compressed BCI Data

This module fine-tunes a language model to process compressed
Speech BCI representations and generate text.
"""

"""
Development Reminders:
    
    GPU Monitoring:
        nvidia-smi -l 5  # Updates every 5 seconds
        nvidia-smi --id=0 --loop=30 --query --display=UTILIZATION
    
    TensorBoard Visualization:
        tensorboard --logdir='/home/ubuntu/speechBCI/data/competitionData/tensorboard/' --port=6008
        # Then open browser to http://localhost:6006/
        
    Monitoring Learning Progres:
            Look at the predicted text and compare to the original text for training set.
        Go to llm_model_dir and look at the predictions files...
        cat MM_LLM_T5_training_set_epoch_7_predictions.txt  | grep "Predicted (original)" | sort | uniq -c
            
            Look at the number of unique words being predicted.
        cat MM_LLM_BART_training_set_epoch_4_predictions.txt | grep "Predict" | grep "original" | sort | uniq -c | \
            awk -F':' '{print $2}' | tr ' ' '\n'| sort | uniq | wc
        cat MM_LLM_BART_training_set_epoch_4_predictions.txt | grep "Predict" | grep "standard" | sort | uniq -c | \
            awk -F':' '{print $2}' | tr ' ' '\n'| sort | uniq | wc
            
    Screen (in an ssh command line session; haven't tried from VSS terminal)
    - screen -S speechBCI_training
    - cd /home/ubuntu/speechBCI
    - source .venv/bin/activate
    - python dev_main_mmllm.py > training_log.txt 2>&1
    - Press Ctrl+A, then press D to detach from the screen.
    - screen -ls
    - screen -r speechBCI_training
    - screen -X -S speechBCI_training quit

"""


import sys
import os
import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import compressed data module instead of embedded
from CompressedDataModule import CompressedBCIDataset, create_data_loaders

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
)

from mmllm.dev_model_utils import create_embedding_model, get_lora_model
from mmllm.label_utils import LabelAnalyzer
from mmllm.dev_training_utils import run_exp

def main():
    """Main function to set up and run the fine-tuning experiment."""
    script_name = "dev_main_mmllm"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***\n")

    # Set device and seed for reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device =", device)
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # Seeds for reproducibility
    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    # ECOG subset
    ecog_subset = "6v_all"  
    
    # LLM configuration
    model_type = "bart"  # or "t5"
    adapter_type = "linear"  # Simpler adapter is sufficient for compressed data
    
    # VQVAE model used to generate the original embeddings that were compressed
    vqvae_model_name = "VQ_VAE_256_512"
    
    # Compressor model name
    compressor_name = f"HC_{vqvae_model_name}"
    
    # Updated experiment name to reflect compressed inputs
    exp_name = f"MM_LLM_COMPRESSED_{model_type.upper()}_{adapter_type.upper()}_{compressor_name}"
    print(f"Experiment name: {exp_name}")
  
    # Directory setup
    root_dir = "/home/ubuntu"
    project_dir = os.path.join(root_dir, "speechBCI")
    data_dir = os.path.join(project_dir, "data/competitionData")

    # Define all directories
    etl_dir = os.path.join(data_dir, "etl", ecog_subset)  
    compress_dir = os.path.join(data_dir, "compressions", compressor_name)  # Use compressed dir instead
    models_base_dir = os.path.join(data_dir, "models")
    tensorboard_base_dir = os.path.join(data_dir, "tensorboard")

    # Model-specific directories
    mmllm_model_dir = os.path.join(models_base_dir, exp_name) 
    os.makedirs(mmllm_model_dir, exist_ok=True)

    # TensorBoard directory
    tensorboard_dir = os.path.join(tensorboard_base_dir, exp_name)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Hyperparameters
    compressed_dim = 256  # Dimension of compressed tokens (from compressor output_dim)
    num_epochs = 10
    learning_rate = 5e-5  # Slightly higher learning rate for compressed data
    training = True
    test_prop = 0.2
    batch_size = 32
    max_gen_seq_len = 64 # Maximum length for generation
    num_gen_beams = 3
    eval_training_set = False

    # Load appropriate model and tokenizer
    if model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
        base_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    elif model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        special_tokens = {"additional_special_tokens": ["<sentence>", "</sentence>"]}
        tokenizer.add_special_tokens(special_tokens)
        base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        base_model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # For label statistics, we need to access the dataset directly
    dataset = CompressedBCIDataset(compress_dir, etl_dir)
    
    # Use the compressed data modules to create data loaders
    # This replaces the earlier dataset creation and splitting logic
    train_dl, test_dl, val_dl = create_data_loaders(
        dataset=dataset,
        batch_size=batch_size,
        test_size=test_prop,
        random_seed=torch_seed
    )
    
    # Add this after creating data loaders in dev_main_mmllm.py
    print("\nVerifying data format...")
    for batch in train_dl:
        compressed_tokens = batch["compressed_tokens"]
        labels = batch["label"]
        print(f"Compressed tokens shape: {compressed_tokens.shape}")
        print(f"Sample token values: {compressed_tokens[0, 0, :5]}")  # First 5 values of first token
        print(f"Number of labels: {len(labels)}")
        print(f"Sample label: '{labels[0]}'")
        break  # Just check one batch

    # Add this to verify shapes
    sample_batch = next(iter(train_dl))
    sample_input = sample_batch["compressed_tokens"]
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Expected shape: [batch_size, seq_length, feature_dim]")
    print(f"Feature dimension: {sample_input.shape[-1]}")
    
    # Analyze labels
    label_stats = LabelAnalyzer(
        dataset.labels,
        dataset.val_flag,
        None,  # No label masks needed for compressed data
        None   # No attention masks needed for compressed data
    )
    label_stats.print_overall_stats()
    label_stats.print_train_test_comparison()

    # Create LoRA model
    lora_base_model = get_lora_model(base_model, model_type=model_type)

    # Create MMLLM model with appropriate adapter
    mm_llm = create_embedding_model(
        model_type=model_type, 
        base_model=lora_base_model, 
        embedding_dim=compressed_dim,  # 256 for your compressed tokens
        adapter_type=adapter_type,
        is_compressed=True  # Add this line to use the new adapters
    )
    mm_llm.to(device)

    # Add this right after creating mm_llm
    print("\nVerifying adapter initialization...")
    if hasattr(mm_llm, "input_adapter"):
        print(f"Adapter type: {type(mm_llm.input_adapter).__name__}")
        # Test forward pass through just the adapter
        with torch.no_grad():
            sample_batch = next(iter(train_dl))
            sample_input = sample_batch["compressed_tokens"].to(device)
            adapted_output = mm_llm.input_adapter(sample_input)
            print(f"Adapter input shape: {sample_input.shape}")
            print(f"Adapter output shape: {adapted_output.shape}")
            print(f"Adapter output sample: {adapted_output[0, 0, :5]}")

    # Display model information
    mm_llm.print_trainable_parameters()

    # Set up optimizer
    optimizer = torch.optim.AdamW(mm_llm.parameters(), lr=learning_rate)

    print(f"\nExperiment Configuration:")
    print(f"- Model Type: {model_type}")
    print(f"- Using Compressed Data: Yes")
    print(f"- Adapter Type: {adapter_type}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Batch Size: {batch_size}")
    print(f"- Num Epochs: {num_epochs}")
    
    # Add this before training
    print("\nTesting full forward pass...")
    sample_batch = next(iter(train_dl))
    sample_inputs = sample_batch["compressed_tokens"].to(device)
    sample_labels = sample_batch["label"]

    # Test forward pass with sample data
    mm_llm.eval()
    with torch.no_grad():
        # Encode labels with max_length from hyperparameters
        encoded_labels = tokenizer(
            sample_labels,
            padding="longest",
            truncation=True,
            max_length=max_gen_seq_len,  # Use hyperparameter instead of hardcoded value
            return_tensors="pt"
        ).to(device)
        
        # Forward pass with and without labels
        outputs = mm_llm(inputs_embeds=sample_inputs)
        print(f"Model output shape (logits): {outputs.logits.shape}")
        
        loss_outputs = mm_llm(
            inputs_embeds=sample_inputs,
            labels=encoded_labels.input_ids
        )
        print(f"Loss: {loss_outputs.loss.item()}")
        
        # Test generation
        try:
            # Get target label for comparison
            target_text = sample_labels[0]
            
            # Generate with proper parameters
            generated = mm_llm.generate(
                inputs_embeds=sample_inputs,
                max_length=max_gen_seq_len,  # Use the hyperparameter instead of hardcoded value
                num_beams=num_gen_beams     # Use the hyperparameter instead of hardcoded value
            )
            
            # Decode the generated output
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            # Process the output according to model type
            from mmllm.dev_training_utils import process_generated_output
            processed_output = process_generated_output(decoded[0], model_type)
            
            # Print both target and generated text for comparison
            print("\nGeneration Test Results:")
            print(f"Target:    '{target_text}'")
            print(f"Generated: '{processed_output}'")
            
            # Calculate simple word overlap for quick assessment
            target_words = set(target_text.lower().split())
            generated_words = set(processed_output.lower().split())
            common_words = target_words.intersection(generated_words)
            
            if len(target_words) > 0:
                overlap_percent = len(common_words) / len(target_words) * 100
                print(f"Word overlap: {len(common_words)}/{len(target_words)} ({overlap_percent:.1f}%)")
            
        except Exception as e:
            print(f"Generation failed: {e}")

    if training:
        # Train and evaluate the model
        trained_model = run_exp(
            exp_name=exp_name,
            train_dl=train_dl,
            test_dl=test_dl,
            val_dl=val_dl,
            model=mm_llm,
            optimizer=optimizer,
            tokenizer=tokenizer,
            device=device,
            num_epochs=num_epochs,
            max_gen_seq_len=max_gen_seq_len,
            num_gen_beams=num_gen_beams,
            model_dir=mmllm_model_dir,
            tensorboard_dir=tensorboard_dir,
            model_type=model_type,
            is_compressed=True,  # Flag for compressed data
            eval_training_set=eval_training_set,
        )

    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"*** {script_name} - END ***")


if __name__ == "__main__":
    main()