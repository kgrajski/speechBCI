"""
Training script for the Hierarchical Attention Compressor.

This script provides standalone training for the compressor using
reconstruction loss, which helps stabilize the full system when integrated.
"""

"""
    Reminder: To monitor GPU utilization, use the following command:
        nvidia-smi --id=0 --loop=30 --query --display=UTILIZATION

    Reminder: To view TensorBoard logs, start TensorBoard on the command line with:
    tensorboard --logdir="/home/ubuntu/speechBCI/data/competitionData/tensorboard/"
    Then open a browser tab to http://localhost:6006/
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm  # Added tqdm import
import torch.nn.functional as F
import plotly.express as px

from dev_SpeechBCIDataSet_Raw import SpeechBCIDataSet_Raw
from dev_HierarchicalCompressor import (
    HierarchicalCompressorWithReconstruction,
    compute_compressor_loss
)


def collate_fn_variable_length(batch):
    """
    Custom collate function that preserves variable-length sequences.
    
    Instead of stacking, it returns a list of tensors with their original shapes.
    """
    # Extract data from batch
    embeddings = [item["vqvae_embeddings"] for item in batch]
    trial_ids = [item.get("trial_id", "") for item in batch]
    labels = [item.get("label", "") for item in batch]
    
    return {
        "vqvae_embeddings": embeddings,
        "trial_ids": trial_ids,
        "labels": labels
    }


def evaluate_compressor(compressor, data_loader, device, alpha=0.1, beta=0.01):
    """
    Evaluate the compressor on a dataset.
    
    Args:
        compressor: HierarchicalCompressorWithReconstruction instance
        data_loader: DataLoader for evaluation
        device: Device to use
        alpha: Weight for diversity loss
        beta: Weight for regularization loss
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, diversity_loss)
    """
    compressor.eval()
    total_loss = 0.0
    recon_loss = 0.0
    diversity_loss = 0.0
    total_samples = 0
    
    # Create progress bar for evaluation
    eval_progress = tqdm(
        total=len(data_loader.dataset), 
        desc="Evaluating",
        leave=False,
        position=1
    )
    
    with torch.no_grad():
        for batch in data_loader:
            # Get list of embeddings (variable length)
            embeddings_list = batch["vqvae_embeddings"]
            batch_size = len(embeddings_list)
            
            # Process each embedding separately with progress tracking
            for emb in embeddings_list:
                # Add batch dimension and move to device
                emb = emb.unsqueeze(0).to(device)
                
                # Compute loss
                loss, r_loss, d_loss = compute_compressor_loss(
                    emb, compressor, alpha=alpha, beta=beta
                )
                
                # Update stats
                total_loss += loss.item()
                recon_loss += r_loss.item()
                diversity_loss += d_loss.item()
                total_samples += 1
                
                # Update progress bar
                eval_progress.update(1)
                eval_progress.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{r_loss.item():.4f}",
                    'div': f"{d_loss.item():.4f}"
                })
    
    # Close progress bar
    eval_progress.close()
    
    # Calculate averages
    if total_samples > 0:
        total_loss /= total_samples
        recon_loss /= total_samples
        diversity_loss /= total_samples
    
    return total_loss, recon_loss, diversity_loss


def train_compressor(
    compressor,
    train_loader,
    test_loader,
    optimizer,
    device,
    num_epochs=10,
    log_dir=None,
    model_dir=None,
    save_name="compressor",
    alpha=0.1,
    beta=0.01
):
    """
    Train the hierarchical compressor with reconstruction loss.
    
    Args:
        compressor: HierarchicalCompressorWithReconstruction instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing during training
        optimizer: Optimizer instance
        device: Device to train on
        num_epochs: Number of training epochs
        log_dir: Directory for TensorBoard logs
        model_dir: Directory for saving models
        save_name: Prefix for saved model files
        alpha: Weight for diversity loss
        beta: Weight for regularization loss
        
    Returns:
        Trained compressor
    """
    # Create TensorBoard writer if log_dir provided
    writer = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    
    # Create save directory if provided
    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)
    
    # Move compressor to device
    compressor = compressor.to(device)
    
    # Track best test loss
    best_test_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        compressor.train()
        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_diversity_loss = 0.0
        train_count = 0
        
        # Add tqdm for batches only
        batch_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in batch_loader:
            # Get list of embeddings (variable length)
            embeddings_list = batch["vqvae_embeddings"]
            batch_size = len(embeddings_list)
            
            # Zero gradients once per batch
            optimizer.zero_grad()
            
            # Accumulate gradients over all examples in the batch
            batch_loss = 0.0
            batch_recon_loss = 0.0
            batch_div_loss = 0.0
            
            for emb in embeddings_list:
                # Add batch dimension and move to device
                emb = emb.unsqueeze(0).to(device)
                
                # Compute loss with batch dimension
                loss, recon_loss, div_loss = compute_compressor_loss(
                    emb, compressor, alpha=alpha, beta=beta
                )
                
                # Scale loss by 1/batch_size to simulate batched behavior
                scaled_loss = loss / batch_size
                
                # Accumulate gradients without stepping
                scaled_loss.backward()
                
                # Update batch statistics
                batch_loss += loss.item()
                batch_recon_loss += recon_loss.item()
                batch_div_loss += div_loss.item()
            
            # Take optimizer step after accumulating gradients
            optimizer.step()
            
            # Update epoch statistics
            train_total_loss += batch_loss
            train_recon_loss += batch_recon_loss
            train_diversity_loss += batch_div_loss
            train_count += batch_size
            
            # Update progress bar with loss information
            batch_loader.set_postfix({
                'loss': f"{batch_loss/batch_size:.4f}",
                'recon': f"{batch_recon_loss/batch_size:.4f}", 
                'div': f"{batch_div_loss/batch_size:.4f}"
            })
        
        # Calculate average training losses
        train_total_loss /= train_count
        train_recon_loss /= train_count
        train_diversity_loss /= train_count
        
        # Testing phase (for model selection and monitoring)
        test_total_loss, test_recon_loss, test_diversity_loss = evaluate_compressor(
            compressor, test_loader, device, alpha, beta
        )
        
        # Print progress
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - {elapsed_time:.2f}s")
        print(f"  Train Loss: {train_total_loss:.6f} (Recon: {train_recon_loss:.6f}, Div: {train_diversity_loss:.6f})")
        print(f"  Test Loss: {test_total_loss:.6f} (Recon: {test_recon_loss:.6f}, Div: {test_diversity_loss:.6f})")
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train_total', train_total_loss, epoch)
            writer.add_scalar('Loss/train_reconstruction', train_recon_loss, epoch)
            writer.add_scalar('Loss/train_diversity', train_diversity_loss, epoch)
            writer.add_scalar('Loss/test_total', test_total_loss, epoch)
            writer.add_scalar('Loss/test_reconstruction', test_recon_loss, epoch)
            writer.add_scalar('Loss/test_diversity', test_diversity_loss, epoch)
        
        # Save best model based on test loss
        if test_total_loss < best_test_loss and model_dir is not None:
            best_test_loss = test_total_loss
            torch.save(
                compressor.state_dict(),
                os.path.join(model_dir, f"{save_name}_best.pt")
            )
            print(f"  Saved best model with test loss: {best_test_loss:.6f}")
        
        # Save checkpoint
        if model_dir is not None and (epoch + 1) % 5 == 0:
            torch.save(
                compressor.state_dict(),
                os.path.join(model_dir, f"{save_name}_epoch{epoch+1}.pt")
            )
    
    # Save final model
    if model_dir is not None:
        torch.save(
            compressor.state_dict(),
            os.path.join(model_dir, f"{save_name}_final.pt")
        )
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    return compressor


def compress_embeddings(compressor, embeddings, device='cuda'):
    """
    Process embeddings through the hierarchical compressor.
    
    Args:
        compressor: Pre-trained HierarchicalCompressorWithReconstruction model
        embeddings: Tensor of shape [time_windows, channels, height, width]
        device: Device to run the compressor on
    
    Returns:
        Compressed tokens of shape [output_tokens, output_dim]
    """
    compressor = compressor.to(device)
    compressor.eval()
    
    with torch.no_grad():
        # Add batch dimension if needed
        if len(embeddings.shape) == 4:  # [T, C, H, W]
            embeddings = embeddings.unsqueeze(0)  # [1, T, C, H, W]
        
        # Move to the correct device
        embeddings = embeddings.to(device)
        
        # Generate compressed representation
        compressed_tokens = compressor(embeddings)  # [1, output_tokens, output_dim]
        
        # Remove batch dimension
        compressed_tokens = compressed_tokens.squeeze(0)  # [output_tokens, output_dim]
    
    return compressed_tokens


def apply_compressor(compressor, data_loader, device, out_dir):
    """
    Compress embeddings and save them.
    
    Args:
        compressor: Pre-trained hierarchical compressor
        data_loader: Input will be one of train_loader, test_loader, or val_loader
        out_dir: Directory to save compressed embeddings
        device: Device to run the compressor on
        out_dir: Directory to save compressed embeddings
    """
    os.makedirs(out_dir, exist_ok=True)
    compressor = compressor.to(device)
    compressor.eval()
    with torch.no_grad():
        # Create progress bar for compression
        compress_progress = tqdm(
            total=len(data_loader.dataset), 
            desc="Compressing",
            leave=False,
            position=1
        )
        
        for batch in data_loader:
            # Get list of embeddings (variable length)
            embeddings_list = batch["vqvae_embeddings"]
            trial_ids = batch["trial_ids"]
            
            for emb, trial_id in zip(embeddings_list, trial_ids):
                # Add batch dimension and move to device
                emb = emb.unsqueeze(0).to(device)
                
                # Compress the embeddings
                compressed_tokens = compress_embeddings(compressor, emb, device)
                
                # Save compressed tokens
                save_path = os.path.join(out_dir, f"{trial_id}.pt")
                torch.save(compressed_tokens.cpu(), save_path)
                
                # Update progress bar
                compress_progress.update(1)
        
        # Close progress bar
        compress_progress.close()
    
    print(f"Compression complete. Compressed tokens saved to {out_dir}")


def main():
    """Main function for training the compressor."""
    # Script identification
    script_name = "dev_train_compressor"
    start_time = time.time()
    print(f"*** {script_name} - START ***\n")
    
        # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set parameters directly (without command-line arguments)
    # ================================================================
    # ECOG subset
    ecog_subset = "6v_all"  # Requirement
    
    # VQVAE model name
    vqvae_model_name = "VQ_VAE_256_512"
    compressor_name = "HC_" + vqvae_model_name
    
    # Directory structure
    root_dir = "/home/ubuntu"
    project_dir = os.path.join(root_dir, "speechBCI")
    data_dir = os.path.join(project_dir, "data/competitionData")
    
    # Data paths
    embed_dir = os.path.join(data_dir, "embeddings", vqvae_model_name)
    etl_dir = os.path.join(data_dir, "etl", ecog_subset)
    compress_dir = os.path.join(data_dir, "compressions", compressor_name)
    
    # Model directories
    models_base_dir = os.path.join(data_dir, "models")
    model_dir = os.path.join(models_base_dir, f"HC_{vqvae_model_name}")
    os.makedirs(model_dir, exist_ok=True)
    
    # TensorBoard directory
    tensorboard_base_dir = os.path.join(data_dir, "tensorboard")
    log_dir = os.path.join(tensorboard_base_dir, f"HC_{vqvae_model_name}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Model configuration
    input_dim = 256         # Dimension of VQ-VAE embeddings
    hidden_dim = 256        # Dimension of hidden representations
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
    # ================================================================
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Indicate whether we are training, embedding, or both
    training = False
    compressing = True

    # Load dataset
    print("Loading dataset...")
    dataset = SpeechBCIDataSet_Raw(
        embed_dir=embed_dir,
        etl_dir=etl_dir
    )
    
    
    # Check if spatial shape is set to auto-detect
    spatial_shape = (spatial_h, spatial_w)
    if spatial_h == 0 or spatial_w == 0:
        detected_shape = dataset.get_spatial_shape()
        if detected_shape is not None:
            spatial_shape = detected_shape
            print(f"Detected spatial shape: {spatial_shape}")
        else:
            spatial_shape = (8, 4)
            print(f"Using default spatial shape: {spatial_shape}")
    
    # Create data splits - validation set is kept completely separate
    train_test_indices = [i for i in range(len(dataset)) 
                        if not dataset.val_flag[i]]
    val_indices = [i for i in range(len(dataset)) 
                if dataset.val_flag[i]]
    
    train_test_dataset = Subset(dataset, train_test_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Split train_test into train and test
    train_size = int((1 - test_prop) * len(train_test_dataset))
    test_size = len(train_test_dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        train_test_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")

    # Create data loaders with variable length handling
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn_variable_length
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn_variable_length
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn_variable_length
    )
    
    # Create compressor model
    print(f"Creating compressor with spatial shape {spatial_shape}...")
    compressor = HierarchicalCompressorWithReconstruction(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        spatial_shape=spatial_shape,
        output_tokens=output_tokens
    )
    
    # Create optimizer
    optimizer = optim.Adam(compressor.parameters(), lr=learning_rate)

    if training:
        # Train compressor using train and test sets
        print("Training compressor...")
        trained_compressor = train_compressor(
            compressor=compressor,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            log_dir=log_dir,
            model_dir=model_dir,
            save_name=compressor_name,
            alpha=alpha,
            beta=beta
        )
        
        # Final evaluation on held-out validation set
        print("\nFinal evaluation on validation set (held-out data):")
        val_total_loss, val_recon_loss, val_diversity_loss = evaluate_compressor(
            trained_compressor, val_loader, device, alpha, beta
        )
        print(f"Validation Loss: {val_total_loss:.6f} (Recon: {val_recon_loss:.6f}, Div: {val_diversity_loss:.6f})")
        print(f"Final model saved to {os.path.join(model_dir, 'hierarchical_compressor_final.pt')}")
    
    if compressing:    
        # Read the best model from the saved state
        print("Loading best compressor model...")
        compressor.load_state_dict(
            torch.load(os.path.join(model_dir, f"{compressor_name}_best.pt"))
        )
        
        # Compress study data
        # Keep in mind how this study handles training, test, and validation data.
        # See above.
        out_dir = os.path.join(compress_dir, "train")
        apply_compressor(compressor, train_loader, device, out_dir)
        apply_compressor(compressor, test_loader, device, out_dir)
        out_dir = os.path.join(compress_dir, "test")
        apply_compressor(compressor, val_loader, device, out_dir)
    
    # Complete
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"*** {script_name} - END ***")


if __name__ == "__main__":
    main()