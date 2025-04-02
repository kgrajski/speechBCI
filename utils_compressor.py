"""
Utility functions for hierarchical compressor training and compression.

This module provides functions for training, evaluating, and using 
the hierarchical compressor model.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import plotly.express as px


def collate_fn_variable_length(batch):
    """
    Custom collate function for batches of variable-length sequences.
    
    Args:
        batch: List of samples from the dataset
    
    Returns:
        Dictionary with batched data
    """
    # Group items by key
    result = {key: [] for key in batch[0].keys()}
    
    for sample in batch:
        for key, value in sample.items():
            result[key].append(value)
    
    return result


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
    
    with torch.no_grad():
        for batch in data_loader:
            # Get list of embeddings (variable length)
            embeddings_list = batch["vqvae_embeddings"]
            batch_size = len(embeddings_list)
            
            # Process each embedding separately
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
    save_dir=None,
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
        save_dir: Directory for saving models
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
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
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
        if test_total_loss < best_test_loss and save_dir is not None:
            best_test_loss = test_total_loss
            torch.save(
                compressor.state_dict(),
                os.path.join(save_dir, f"{save_name}_best.pt")
            )
            print(f"  Saved best model with test loss: {best_test_loss:.6f}")
        
        # Save checkpoint
        if save_dir is not None and (epoch + 1) % 5 == 0:
            torch.save(
                compressor.state_dict(),
                os.path.join(save_dir, f"{save_name}_epoch{epoch+1}.pt")
            )
    
    # Save final model
    if save_dir is not None:
        torch.save(
            compressor.state_dict(),
            os.path.join(save_dir, f"{save_name}_final.pt")
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


def compress_study_data(compressor, dataset, device, compress_dir):
    """
    Compress all study data embeddings and save them.
    
    Args:
        compressor: Pre-trained hierarchical compressor
        dataset: The dataset containing VQ-VAE embeddings
        device: Device to run the compressor on
        compress_dir: Directory to save compressed embeddings
    """
    # Create the subdirectories for train and test compressed data
    train_dir = os.path.join(compress_dir, "train")
    test_dir = os.path.join(compress_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get the set of unique sample idkeys in the dataset
    sample_idkeys = set(dataset.sample_idkey)
    
    # Statistics for compressed representations
    max_series_len = 0
    all_compressed_stats = []
    
    # Process each idkey
    for idkey in tqdm(sample_idkeys, desc="Compressing samples"):
        # Find all indices for this idkey
        indices = [i for i in range(len(dataset)) if dataset.sample_idkey[i] == idkey]
        indices = sorted(indices)  # Ensure the indices are in order
        
        # Get the embeddings for this idkey
        embeddings_list = [dataset[i]["vqvae_embeddings"] for i in indices]
        
        # Stack them along the time dimension
        embeddings = torch.stack(embeddings_list, dim=0)
        
        # Compress the embeddings
        compressed = compress_embeddings(compressor, embeddings, device)
        
        # Determine the output directory (train or test)
        output_dir = test_dir if dataset.val_flag[indices[0]] else train_dir
        
        # Save the compressed representation
        filename = os.path.join(output_dir, f"{idkey}.pt")
        torch.save(compressed, filename)
        
        # Collect statistics
        if len(indices) > max_series_len:
            max_series_len = len(indices)
        
        # Collect stats on compressed tokens for visualization
        token_stats = torch.mean(compressed, dim=0).cpu().numpy()
        all_compressed_stats.append(token_stats)
    
    print(f"Maximum series length: {max_series_len}")
    
    # Generate and save visualization of compressed token statistics
    all_stats = np.stack(all_compressed_stats)
    
    # Create a heatmap of token statistics across samples
    fig = px.imshow(
        all_stats, 
        labels=dict(x="Output Dimension", y="Sample Index", color="Value"),
        title="Compressed Token Statistics Across Samples"
    )
    fig.write_html(os.path.join(compress_dir, "compressed_stats.html"))
    
    # Create a histogram of mean token values
    mean_values = all_stats.flatten()
    fig = px.histogram(
        mean_values, 
        title="Distribution of Compressed Token Values",
        labels={'x': 'Token Value', 'y': 'Count'}
    )
    fig.write_html(os.path.join(compress_dir, "token_distribution.html"))
    
    print(f"Compression complete. Compressed tokens saved to {compress_dir}")


# Need to also include compute_compressor_loss since it's used by the functions above
def compute_compressor_loss(
    original_sequence,
    compressor,
    alpha=0.1,
    beta=0.01,
    reduction='mean'
):
    """
    Compute combined loss for compressor training.
    
    Args:
        original_sequence: Tensor of shape [batch, time_windows, channels, height, width]
                          with typical dimensions [B, T, 256, 8, 4]
        compressor: HierarchicalCompressorWithReconstruction instance
        alpha: Weight for diversity loss term
        beta: Weight for regularization loss term
        reduction: Reduction method for losses ('mean', 'sum', 'none')
    
    Returns:
        Tuple of (total_loss, recon_loss, diversity_loss):
            - total_loss: Combined weighted loss
            - recon_loss: Reconstruction loss (MSE) on channel dimension only
            - diversity_loss: Cosine similarity loss between output tokens
    """
    # Get exact dimensions
    batch_size, time_windows, channels, height, width = original_sequence.shape
    
    # Flatten input to the shape expected by the compressor
    flattened_input = original_sequence.reshape(batch_size, time_windows, channels * height * width)
    
    # Forward pass with reconstruction
    compressed, reconstructed, num_windows = compressor(
        flattened_input, with_reconstruction=True
    )
    
    # Extract channel dimension for reconstruction comparison
    # The model only reconstructs the channels (256) not the full C*H*W (8192)
    # Take the first element of each spatial position
    original_channels = original_sequence[:, :num_windows, :, 0, 0]
    
    # Compute reconstruction loss comparing just the channel dimension
    recon_loss = F.mse_loss(reconstructed, original_channels, reduction=reduction)
    
    # Compute diversity loss
    tokens = compressed.view(-1, compressed.size(-1))
    norm_tokens = F.normalize(tokens, p=2, dim=1)
    cosine_sim = torch.matmul(norm_tokens, norm_tokens.transpose(0, 1))
    mask = torch.eye(cosine_sim.size(0), device=cosine_sim.device)
    diversity_loss = (cosine_sim * (1.0 - mask)).mean()
    
    # Compute regularization loss
    reg_loss = beta * tokens.pow(2).mean()
    
    # Combine losses
    total_loss = recon_loss + alpha * diversity_loss + reg_loss
    
    return total_loss, recon_loss, diversity_loss