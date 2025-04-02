"""
Multimodal Language Model with Hierarchically Compressed BCI Data

This module fine-tunes a language model to process compressed
Speech BCI representations and generate text.
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

from mmllm.model_utils import create_embedding_model, get_lora_model
from mmllm.label_utils import LabelAnalyzer
from mmllm.training_utils import run_exp

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
    compress_dir = os.path.join(data_dir, "compressed", compressor_name)  # Use compressed dir instead
    models_base_dir = os.path.join(data_dir, "models")
    tensorboard_base_dir = os.path.join(data_dir, "tensorboard")

    # Model-specific directories
    mmllm_model_dir = os.path.join(models_base_dir, exp_name) 
    os.makedirs(mmllm_model_dir, exist_ok=True)

    # TensorBoard directory
    tensorboard_dir = os.path.join(tensorboard_base_dir, exp_name)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Hyperparameters
    compressed_dim = 512  # Dimension of compressed tokens (from compressor output_dim)
    num_epochs = 10
    learning_rate = 5e-5  # Slightly higher learning rate for compressed data
    training = True
    test_prop = 0.2
    batch_size = 16
    max_gen_seq_len = 64
    num_gen_beams = 3
    eval_training_set = True

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

    # Use the compressed data modules to create data loaders
    # This replaces the earlier dataset creation and splitting logic
    train_dl, test_dl, val_dl = create_data_loaders(
        compress_dir=compress_dir,
        etl_dir=etl_dir,
        batch_size=batch_size,
        test_size=test_prop,
        random_seed=torch_seed
    )
    
    # For label statistics, we need to access the dataset directly
    dataset = CompressedBCIDataset(compress_dir, etl_dir)
    
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
        embedding_dim=compressed_dim,
        adapter_type=adapter_type,
        is_compressed=True  # New flag to indicate compressed data
    )

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