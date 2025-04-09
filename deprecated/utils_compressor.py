"""
Utility functions for hierarchical compressor training and compression.

This module provides functions for training, evaluating, and using 
the hierarchical compressor model.
"""

import os
import time
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
        and dictionary of token statistics
    """
    compressor.eval()
    total_loss = 0.0
    recon_loss = 0.0
    diversity_loss = 0.0
    total_samples = 0
    
    # Track token statistics across evaluation
    token_stats_eval = None
    token_indices_all = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Get list of embeddings (variable length)
            embeddings_list = batch["vqvae_embeddings"]
            
            # Process each embedding separately
            for emb in embeddings_list:
                # Add batch dimension and move to device
                emb = emb.unsqueeze(0).to(device)
                
                # Get outputs directly from compressor
                outputs = compressor(emb, with_reconstruction=True)
                
                # Get losses
                r_loss = outputs['recon_loss']
                # Use enhanced diversity loss if available
                d_loss = outputs.get('enhanced_div_loss', outputs['diversity_loss'])
                reg_loss = outputs['reg_loss']
                
                # Calculate total loss
                loss = r_loss + alpha * d_loss + beta * reg_loss
                
                # Update stats
                total_loss += loss.item()
                recon_loss += r_loss.item()
                diversity_loss += d_loss.item()
                total_samples += 1
                
                # Collect token indices if available
                if 'token_indices' in outputs:
                    token_indices_all.append(outputs['token_indices'].cpu())
                
                # Collect token stats
                if 'token_stats' in outputs:
                    if token_stats_eval is None:
                        token_stats_eval = {k: 0 for k in outputs['token_stats']}
                    for k, v in outputs['token_stats'].items():
                        if isinstance(v, (int, float)):
                            token_stats_eval[k] += v / total_samples
    
    # Calculate averages
    if total_samples > 0:
        total_loss /= total_samples
        recon_loss /= total_samples
        diversity_loss /= total_samples
    
    # Return token usage statistics along with losses
    stats = {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'diversity_loss': diversity_loss,
        'token_stats': token_stats_eval
    }
    
    # Return the traditional tuple for backward compatibility
    return total_loss, recon_loss, diversity_loss, stats


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
    beta=0.01,
    patience=10
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
        patience: Number of epochs to wait for improvement before early stopping
        
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
    
    # Track best model and early stopping
    best_model_state = None
    best_loss = float('inf')
    epochs_no_improve = 0
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, 
        min_lr=1e-6, verbose=True
    )
    
    # Training loop
    start_training_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        compressor.train()
        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_diversity_loss = 0.0
        train_count = 0
        token_stats_epoch = None  # Will hold accumulated token stats
        
        # Add tqdm for batches
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
            batch_indices = []  # Collect token indices for visualization
            
            for emb in embeddings_list:
                # Add batch dimension and move to device
                emb = emb.unsqueeze(0).to(device)
                
                # Forward pass with reconstruction and enhanced diversity loss
                outputs = compressor(emb, with_reconstruction=True)
                
                # Get losses
                recon_loss = outputs['recon_loss']
                div_loss = outputs.get('enhanced_div_loss', outputs['diversity_loss'])
                reg_loss = outputs['reg_loss']
                
                # Calculate total loss with weights
                loss = recon_loss + alpha * div_loss + beta * reg_loss
                
                # Scale loss by 1/batch_size to simulate batched behavior
                scaled_loss = loss / batch_size
                
                # Accumulate gradients without stepping
                scaled_loss.backward()
                
                # Update batch statistics
                batch_loss += loss.item()
                batch_recon_loss += recon_loss.item()
                batch_div_loss += div_loss.item()
                
                # Collect token indices if available
                if 'token_indices' in outputs:
                    batch_indices.append(outputs['token_indices'].detach().cpu())
                
                # Collect token stats for visualization
                if 'token_stats' in outputs:
                    if token_stats_epoch is None:
                        token_stats_epoch = {k: 0 for k in outputs['token_stats']}
                    for k, v in outputs['token_stats'].items():
                        if isinstance(v, (int, float)):
                            token_stats_epoch[k] += v / batch_size
            
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
        test_total_loss, test_recon_loss, test_diversity_loss, test_stats = evaluate_compressor(
            compressor, test_loader, device, alpha, beta
        )
        
        # Print progress
        elapsed_time = time.time() - epoch_start_time
        total_time = time.time() - start_training_time
        print(f"Epoch {epoch+1}/{num_epochs} - {elapsed_time:.2f}s (Total: {total_time:.2f}s)")
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
            
            # Log token usage statistics if available
            if token_stats_epoch:
                for k, v in token_stats_epoch.items():
                    writer.add_scalar(f'Tokens/{k}', v, epoch)
                
                # Every 5 epochs, log token usage histogram
                if epoch % 5 == 0 and batch_indices:
                    all_indices = torch.cat([idx.flatten() for idx in batch_indices])
                    writer.add_histogram('Tokens/usage_distribution', all_indices, epoch)
        
        # Save best model
        if test_total_loss < best_loss:
            best_loss = test_total_loss
            best_model_state = compressor.state_dict()
            print(f"  Saving best model (Loss: {best_loss:.6f})")
            torch.save(best_model_state, os.path.join(save_dir, f"{save_name}_best.pt"))
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")
        
        # Step the scheduler
        scheduler.step(test_total_loss)
        
        # Early stopping check
        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs without improvement")
            break
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    # Load best model
    if best_model_state is not None:
        compressor.load_state_dict(best_model_state)
    
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


def compute_token_diversity_loss(token_usage):
    """
    Compute token diversity loss based on token usage distribution.
    Higher perplexity (more uniform distribution) is better.
    
    Args:
        token_usage: Tensor of shape [batch_size, num_tokens]
            containing counts of each token's usage
    
    Returns:
        Tensor: The negative log perplexity (lower is better)
    """
    # Normalize to get probability distribution
    probs = token_usage / (torch.sum(token_usage, dim=1, keepdim=True) + 1e-10)
    
    # Calculate entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    
    # Calculate perplexity (2^entropy)
    perplexity = torch.exp(entropy)
    
    # Return negative log perplexity as loss (lower is better)
    return -torch.log(perplexity + 1e-10).mean()
