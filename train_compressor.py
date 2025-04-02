"""
Training script for the Hierarchical Attention Compressor.

This script provides standalone training for the compressor using
reconstruction loss, which helps stabilize the full system when integrated.
"""

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
import torch.optim as optim

from SpeechBCIDataSet_Raw import SpeechBCIDataSet_Raw
from HierarchicalCompressor import HierarchicalCompressorWithReconstruction
from utils_compressor import (
    collate_fn_variable_length,
    train_compressor,
    compress_study_data
)


def main():
    """Main function for training the compressor and compressing embeddings."""
    # Script identification
    script_name = "train_compressor"
    start_time = time.time()
    print(f"*** {script_name} - START ***\n")
    
    # Set parameters directly (without command-line arguments)
    # ================================================================
    # ECOG subset
    ecog_subset = "6v_all"  # Requirement
    
    # VQVAE model name
    vqvae_model_name = "VQ_VAE_256_512"
    
    # Directory structure
    root_dir = "/home/ubuntu"
    project_dir = os.path.join(root_dir, "speechBCI")
    data_dir = os.path.join(project_dir, "data/competitionData")
    
    # Data paths
    embed_dir = os.path.join(data_dir, "embeddings", vqvae_model_name)
    etl_dir = os.path.join(data_dir, "etl", ecog_subset)
    
    # Model directories
    models_base_dir = os.path.join(data_dir, "models")
    compressor_name = "HC_" + vqvae_model_name
    save_dir = os.path.join(models_base_dir, compressor_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard directory
    tensorboard_base_dir = os.path.join(data_dir, "tensorboard")
    log_dir = os.path.join(tensorboard_base_dir, compressor_name)
    
    # Compression directory
    compress_base_dir = os.path.join(data_dir, "compressed")
    compress_dir = os.path.join(compress_base_dir, compressor_name)
    os.makedirs(compress_dir, exist_ok=True)
    
    # Model configuration
    input_dim = 256         # Dimension of VQ-VAE embeddings
    hidden_dim = 256        # Dimension of hidden representations
    output_dim = 512        # Dimension of output tokens
    output_tokens = 512     # Number of output tokens
    spatial_h = 8           # Height of spatial grid
    spatial_w = 4           # Width of spatial grid
    
    # Training configuration
    batch_size = 16         # Batch size for training
    num_epochs = 50         # Number of training epochs
    learning_rate = 1e-4    # Learning rate
    alpha = 0.1             # Weight for diversity loss
    beta = 0.01             # Weight for regularization loss
    test_prop = 0.2         # Proportion of non-validation data for testing
    
    # Random seed
    seed = 42
    
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Indicate whether we are training, compressing, or both
    training = True
    compressing = True
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ================================================================
    
    # Dataset and DataLoader setup
    dataset = SpeechBCIDataSet_Raw(
        ecog_subset=ecog_subset, 
        vqvae_model_name=vqvae_model_name
    )
    
    # Split into test and training
    val_indices = [i for i, flag in enumerate(dataset.val_flag) if flag]
    train_indices = [i for i, flag in enumerate(dataset.val_flag) if not flag]
    
    # Further split training into train and test
    train_size = int((1 - test_prop) * len(train_indices))
    test_size = len(train_indices) - train_size
    
    # Create train/test split
    generator = torch.Generator().manual_seed(seed)
    train_subset, test_subset = random_split(
        train_indices, [train_size, test_size], generator=generator
    )
    
    # Create actual train and test datasets
    train_dataset = Subset(dataset, list(train_subset))
    test_dataset = Subset(dataset, list(test_subset))
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Dataset splits: Train={len(train_dataset)}, Test={len(test_dataset)}, Val={len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn_variable_length
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        collate_fn=collate_fn_variable_length
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        collate_fn=collate_fn_variable_length
    )
    
    if training:
        # Create the compressor model
        compressor = HierarchicalCompressorWithReconstruction(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_tokens=output_tokens,
            spatial_shape=[spatial_h, spatial_w],
            max_input_windows=250,
            num_layers=4,
            num_heads=8
        )
        
        # Create optimizer
        optimizer = optim.Adam(compressor.parameters(), lr=learning_rate)
        
        # Train the compressor
        print("Starting compressor training...")
        trained_compressor = train_compressor(
            compressor,
            train_loader,
            test_loader,
            optimizer,
            device,
            num_epochs=num_epochs,
            log_dir=log_dir,
            save_dir=save_dir,
            save_name=compressor_name,
            alpha=alpha,
            beta=beta
        )
        print("Compressor training completed.")
    else:
        # Load pre-trained compressor
        print(f"Loading pre-trained compressor from {save_dir}...")
        compressor = HierarchicalCompressorWithReconstruction(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_tokens=output_tokens,
            spatial_shape=[spatial_h, spatial_w],
            max_input_windows=250,
            num_layers=4,
            num_heads=8
        )
        compressor.load_state_dict(
            torch.load(os.path.join(save_dir, f"{compressor_name}_best.pt"))
        )
        trained_compressor = compressor
    
    if compressing:
        # Compress the study data
        print(f"Compressing study data with trained compressor...")
        compress_study_data(trained_compressor, dataset, device, compress_dir)
        print("Compression completed.")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal runtime: {int(hours):02}h {int(minutes):02}m {seconds:.2f}s")
    
    print(f"*** {script_name} - END ***")


if __name__ == "__main__":
    main()