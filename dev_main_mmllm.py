"""
Main script for the multimodal language model with hierarchical compression.

This script integrates the hierarchical compressor with the language model adapter,
providing a complete pipeline for processing neural data and generating text.
"""

import os
import time
import gc
import argparse
import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration

# Import existing components
from main_mmllm import create_embedding_model, run_exp, get_lora_model, get_vqvae_codebook_average

# Import new components for hierarchical compression
from dev_SpeechBCIDataSet_Raw import SpeechBCIDataSet_Raw
from dev_HierarchicalCompressor import HierarchicalAttentionCompressor
#from dev_CompressedDataModule import create_data_loaders


def main():
    """Main function for the Speech BCI project with hierarchical compression."""
    # Script identification
    script_name = "dev_main_mmllm_compressed"
    start_time = time.perf_counter()
    print(f"*** {script_name} - START ***\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Set random seeds
    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    
    # Configuration
    # ECOG subset
    ecog_subset = "6v_all"  # Requirement
    
    # Choose model type: 't5' or 'bart'
    model_type = "bart"  # Change to 't5' to use T5 instead
    
    # Choose adapter type: 'linear', 'lstm', 'conv', 'attention', or 'rnn'
    adapter_type = "linear"  # Change to appropriate adapter
    
    # Attention configuration (if using attention adapter)
    attention_mode = "global"  # Options: 'global', 'causal', 'local'
    window_size = None  # For local attention window size
    
    # Hierarchical compressor configuration
    compression_enabled = True  # Set to False to disable compression
    compression_hidden_dim = 256
    compression_output_tokens = 512
    spatial_shape = (8, 4)  # 8×4 grid
    pretrained_compressor = True  # Use pretrained compressor
    
    # Experiment name
    vqvae_model_name = "VQVAE_4C_16H_8W_128_256"
    compressor_suffix = "_HC" if compression_enabled else ""
    exp_name = f"MM_LLM_{model_type.upper()}_{adapter_type.upper()}_{vqvae_model_name}{compressor_suffix}"
    print(f"Experiment name: {exp_name}")
    
    # Directory setup
    root_dir = "/home/ubuntu"
    project_dir = os.path.join(root_dir, "speechBCI")
    data_dir = os.path.join(project_dir, "data/competitionData")
    
    # Define all data directories
    etl_dir = os.path.join(data_dir, "etl", ecog_subset)
    embed_dir = os.path.join(data_dir, "embeddings")
    models_base_dir = os.path.join(data_dir, "models")
    tensorboard_base_dir = os.path.join(data_dir, "tensorboard")
    
    # Model-specific directories
    vqvae_model_dir = os.path.join(models_base_dir, vqvae_model_name)
    embed_dir = os.path.join(embed_dir, vqvae_model_name)
    mmllm_model_dir = os.path.join(models_base_dir, exp_name)
    os.makedirs(mmllm_model_dir, exist_ok=True)
    
    # Compressor directory
    compressor_model_dir = os.path.join(models_base_dir, f"HC_{vqvae_model_name}")
    compressor_model_file = os.path.join(compressor_model_dir, "hierarchical_compressor_best.pt")
    
    # TensorBoard directory
    tensorboard_dir = os.path.join(tensorboard_base_dir, exp_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Hyperparameters
    embedding_dim = 64  # Dimension of VQ-VAE embeddings
    num_embeddings = 256  # Size of VQ-VAE codebook
    
    max_seq_len = 512  # Maximum sequence length for labels
    num_epochs = 5
    learning_rate = 1e-5
    training = True
    test_prop = 0.2
    train_prop = 1 - test_prop
    batch_size = 16
    max_gen_seq_len = 64
    num_gen_beams = 3
    eval_training_set = True
    
    # ==================== Language Model Setup ====================
    # Load appropriate model and tokenizer based on model type
    if model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
        base_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    elif model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        # Add special tokens for sentence boundaries
        special_tokens = {"additional_special_tokens": ["<sentence>", "</sentence>"]}
        tokenizer.add_special_tokens(special_tokens)
        base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        base_model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # ==================== Data Loading ====================
    # Load raw dataset
    print("Loading raw dataset...")
    raw_dataset = SpeechBCIDataSet_Raw(
        embed_dir=embed_dir,
        etl_dir=etl_dir
    )
    
    # ==================== Compressor Setup ====================
    if compression_enabled:
        print("Setting up hierarchical compressor...")
        
        # Create compressor
        compressor = HierarchicalAttentionCompressor(
            input_dim=embedding_dim,
            hidden_dim=compression_hidden_dim,
            spatial_shape=spatial_shape,
            output_tokens=compression_output_tokens
        )
        
        # Load pretrained weights if available
        if pretrained_compressor and os.path.exists(compressor_model_file):
            print(f"Loading pretrained compressor from {compressor_model_file}")
            compressor.load_state_dict(torch.load(compressor_model_file))
        else:
            print("Using randomly initialized compressor")
        
        # Create data loaders with compression
        print("Creating compressed data loaders...")
        train_dl, test_dl, val_dl = create_data_loaders(
            raw_dataset=raw_dataset,
            compressor=compressor,
            tokenizer=tokenizer,
            model_type=model_type,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            test_prop=test_prop,
            device=device,
            seed=torch_seed
        )
    else:
        # Load VQVAE for padding vector
        from models.vqvae import VQVAE
        vqvae_model = VQVAE(4, 128, embedding_dim, num_embeddings)
        vqvae_model.load_state_dict(torch.load(os.path.join(vqvae_model_dir, vqvae_model_name + "_final.pt")))
        padding_vector = get_vqvae_codebook_average(vqvae_model)
        
        # Use original dataset and data loaders
        from SpeechBCIDataSet_Embedded import SpeechBCIDataSet_Embedded
        from torch.utils.data import DataLoader, Subset, random_split
        
        print("Using original dataset without compression...")
        study_dataset = SpeechBCIDataSet_Embedded(
            embed_dir=embed_dir,
            etl_dir=etl_dir,
            tokenizer=tokenizer,
            model_type=model_type,
            max_seq_len=max_seq_len,
            padding_vector=padding_vector,
        )
        
        # Create dataset splits
        train_test_indices = [i for i in range(len(study_dataset.val_flag)) if study_dataset.val_flag[i] is False]
        val_indices = [i for i in range(len(study_dataset.val_flag)) if study_dataset.val_flag[i] is True]
        train_test_dataset = Subset(study_dataset, train_test_indices)
        train_dataset, test_dataset = random_split(train_test_dataset, [train_prop, test_prop])
        val_dataset = Subset(study_dataset, val_indices)
        
        # Create dataloaders
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ==================== Model Setup ====================
    # Create LoRA model
    lora_base_model = get_lora_model(base_model, model_type=model_type)
    
    # Create MMLLM model with adapter
    mm_llm = create_embedding_model(
        model_type=model_type,
        base_model=lora_base_model,
        embedding_dim=embedding_dim,  # Same dimension regardless of compression
        adapter_type=adapter_type,
        attention_mode=attention_mode,
        window_size=window_size
    )
    
    # Print model information
    mm_llm.print_trainable_parameters()
    
    # Set up optimizer
    if compression_enabled and training:
        # Include compressor parameters in training
        optimizer = torch.optim.AdamW(
            list(mm_llm.parameters()) + list(compressor.parameters()),
            lr=learning_rate
        )
    else:
        # Only train adapter parameters
        optimizer = torch.optim.AdamW(mm_llm.parameters(), lr=learning_rate)
    
    # ==================== Training ====================
    if training:
        print("\nStarting training...")
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
            eval_training_set=eval_training_set,
        )
        
        # Save compressor if used
        if compression_enabled:
            torch.save(
                compressor.state_dict(),
                os.path.join(mmllm_model_dir, "compressor_final.pt")
            )
    
    # ==================== Complete ====================
    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"*** {script_name} - END ***")


if __name__ == "__main__":
    main()