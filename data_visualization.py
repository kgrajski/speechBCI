"""
BCI Data Visualization Tool - Per-Trial UMAP Edition

This script creates UMAP visualizations of raw embedded data and compressed tokens,
with each trial independently mapped to its own UMAP space to reduce memory usage.

Features:
- Individual UMAP projection for each trial
- Side-by-side comparison of embedded and compressed representations
- Interactive Plotly Express visualizations
- Filenames include word count for easy classification
- Memory-efficient processing
"""

import os
import numpy as np
import torch
import pandas as pd
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from datetime import datetime
import warnings
import re

# Override PyTorch's default loader behavior to always load to CPU
_original_load = torch.load
torch.load = lambda f, *args, **kwargs: _original_load(f, map_location=torch.device('cpu'), *args, **kwargs)

# Force PyTorch to use CPU
torch.set_default_tensor_type('torch.FloatTensor')  # Use CPU by default
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU from PyTorch

# Silence specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import project-specific modules
from SpeechBCIDataSet_Raw import SpeechBCIDataSet_Raw
from CompressedDataModule import CompressedBCIDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def setup_directories():
    """Set up directories for data and output."""
    # Directory setup
    root_dir = "/home/ubuntu"
    project_dir = os.path.join(root_dir, "speechBCI")
    data_dir = os.path.join(project_dir, "data/competitionData")

    # VQVAE model used for original embeddings
    vqvae_model_name = "VQ_VAE_256_512"
    compressor_name = f"HC_{vqvae_model_name}"

    # ECOG subset
    ecog_subset = "6v_all"

    # Define directories
    etl_dir = os.path.join(data_dir, "etl", ecog_subset)
    embedded_dir = os.path.join(data_dir, "embeddings", vqvae_model_name)
    compressed_dir = os.path.join(data_dir, "compressions", compressor_name)

    # Output directory for saved visualizations
    output_dir = os.path.join(data_dir, "visualizations", "trajectory_analysis", vqvae_model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Embedded data directory: {embedded_dir}")
    print(f"Compressed data directory: {compressed_dir}")
    print(f"Output directory: {output_dir}")
    
    return etl_dir, embedded_dir, compressed_dir, output_dir


def load_data(embedded_dir, etl_dir, compressed_dir=None):
    """Load datasets and create train/test/val splits matching CompressedDataModule logic."""
    print("Loading embedded data...")
    embedded_dataset = SpeechBCIDataSet_Raw(
        embed_dir=embedded_dir,
        etl_dir=etl_dir
    )
    
    print(f"Loaded {len(embedded_dataset)} embedded examples")
    
    # Load compressed data if available
    compressed_dataset = None
    if compressed_dir and os.path.exists(compressed_dir):
        print("Loading compressed data...")
        compressed_dataset = CompressedBCIDataset(
            compress_dir=compressed_dir,
            etl_dir=etl_dir
        )
        print(f"Loaded {len(compressed_dataset)} compressed examples")
    
    # Create a mapping of trial_ids to their train/test/val status using compressed dataset's logic
    # If we don't have compressed data, we'll create our own split
    trial_split_map = {}
    
    if compressed_dataset:
        # Use compressed dataset's split designation
        print("Using compressed dataset's train/test/validation split for consistency")
        for i in range(len(compressed_dataset)):
            item = compressed_dataset[i]
            trial_id = item["trial_id"]
            is_val = item["is_validation"]
            trial_split_map[trial_id] = "val" if is_val else "train_test"
    
    # Split the embedded dataset based on trial_id mappings or create a new split
    train_embedded_indices = []
    test_embedded_indices = []
    val_embedded_indices = []
    
    # Track trial_ids we've already processed to avoid duplicates
    processed_trial_ids = set()
    
    for i in range(len(embedded_dataset)):
        sample = embedded_dataset[i]
        trial_id = sample["trial_id"]
        
        # Skip duplicates if they exist
        if trial_id in processed_trial_ids:
            continue
        
        processed_trial_ids.add(trial_id)
        
        if trial_id in trial_split_map:
            # Use the mapping from compressed dataset
            split = trial_split_map[trial_id]
            if split == "val":
                val_embedded_indices.append(i)
            else:
                train_embedded_indices.append(i)
        else:
            # Create our own split
            if len(train_embedded_indices) < int(0.8 * len(embedded_dataset)):
                train_embedded_indices.append(i)
            else:
                test_embedded_indices.append(i)
    
    print(f"Train set: {len(train_embedded_indices)} examples")
    print(f"Test set: {len(test_embedded_indices)} examples")
    print(f"Validation set: {len(val_embedded_indices)} examples")
    
    # Create data loaders
    train_embedded_loader = DataLoader(
        torch.utils.data.Subset(embedded_dataset, train_embedded_indices), 
        batch_size=1, shuffle=False
    )
    test_embedded_loader = DataLoader(
        torch.utils.data.Subset(embedded_dataset, test_embedded_indices), 
        batch_size=1, shuffle=False
    )
    val_embedded_loader = DataLoader(
        torch.utils.data.Subset(embedded_dataset, val_embedded_indices), 
        batch_size=1, shuffle=False
    )
    
    train_compressed_loader = None
    test_compressed_loader = None
    val_compressed_loader = None
    
    if compressed_dataset:
        train_compressed_indices = []
        test_compressed_indices = []
        val_compressed_indices = []
        
        for i in range(len(compressed_dataset)):
            item = compressed_dataset[i]
            trial_id = item["trial_id"]
            if trial_id in trial_split_map:
                split = trial_split_map[trial_id]
                if split == "val":
                    val_compressed_indices.append(i)
                else:
                    train_compressed_indices.append(i)
            else:
                test_compressed_indices.append(i)
        
        train_compressed_loader = DataLoader(
            torch.utils.data.Subset(compressed_dataset, train_compressed_indices), 
            batch_size=1, shuffle=False
        )
        test_compressed_loader = DataLoader(
            torch.utils.data.Subset(compressed_dataset, test_compressed_indices), 
            batch_size=1, shuffle=False
        )
        val_compressed_loader = DataLoader(
            torch.utils.data.Subset(compressed_dataset, val_compressed_indices), 
            batch_size=1, shuffle=False
        )
    
    return (train_embedded_loader, test_embedded_loader, val_embedded_loader,
            train_compressed_loader, test_compressed_loader, val_compressed_loader,
            embedded_dataset, compressed_dataset)


def get_umap_model():
    """Create a new UMAP model with standard parameters."""
    return umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )


def create_embedded_plot(embedded_data, label, trial_id):
    """Create a UMAP visualization of embedded data with trajectory."""
    # Create and fit UMAP model for this specific trial
    print(f"Fitting UMAP for embedded data (trial {trial_id})...")
    umap_model = get_umap_model()
    
    # Convert to numpy for UMAP
    embedded_numpy = embedded_data.detach().cpu().numpy()
    
    # Reshape if needed
    if len(embedded_numpy.shape) > 2:
        T = embedded_numpy.shape[0]
        # Flatten all dimensions except the first (time)
        embedded_numpy = embedded_numpy.reshape(T, -1)

    print(f"UMAP input shape: {embedded_numpy.shape}")  # Debug info

    # Fit and transform data
    umap_model.fit(embedded_numpy)
    embedding = umap_model.transform(embedded_numpy)
    
    # Create a DataFrame for Plotly
    n_points = embedding.shape[0]
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'time': range(n_points),
        'step': range(n_points)
    })
    
    # Create figure
    fig = go.Figure()
    
    # Add line for trajectory
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='lines',
        line=dict(
            color='rgba(100,100,100,0.5)',
            width=1
        ),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            color=df['time'],
            colorscale='Viridis',
            size=6,
            opacity=0.8,
            showscale=True,
            colorbar=dict(
                title='Time'
            )
        ),
        text=[f"Step: {i}" for i in range(n_points)],
        hoverinfo='text',
        showlegend=False
    ))
    
    # Add start point (first step)
    fig.add_trace(go.Scatter(
        x=[embedding[0, 0]],
        y=[embedding[0, 1]],
        mode='markers',
        marker=dict(
            color='green',
            size=12,
            line=dict(color='black', width=1)
        ),
        name='Start',
        showlegend=True
    ))
    
    # Add end point (last step)
    fig.add_trace(go.Scatter(
        x=[embedding[-1, 0]],
        y=[embedding[-1, 1]],
        mode='markers',
        marker=dict(
            color='red',
            size=12,
            line=dict(color='black', width=1)
        ),
        name='End',
        showlegend=True
    ))
    
    # Add arrows to show direction
    for i in range(0, n_points-1, max(1, n_points//10)):  # Add arrows at several points
        fig.add_annotation(
            x=embedding[i+1, 0],
            y=embedding[i+1, 1],
            ax=embedding[i, 0],
            ay=embedding[i, 1],
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='black'
        )
    
    # Update layout
    fig.update_layout(
        title=f"Trial ID: {trial_id}<br>Label: '{label}'",
        showlegend=True,
        width=800,
        height=600,
        xaxis=dict(
            title=None,
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title=None,
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        hovermode='closest'
    )
    
    return fig


def create_compressed_plot(compressed_tokens, label, trial_id):
    """Create a UMAP visualization of compressed tokens with trajectory."""
    # Create and fit UMAP model for this specific trial
    print(f"Fitting UMAP for compressed data (trial {trial_id})...")
    umap_model = get_umap_model()
    
    # Convert to numpy for UMAP
    data_numpy = compressed_tokens.detach().cpu().numpy()
    
    # Fit and transform data
    umap_model.fit(data_numpy)
    embedding = umap_model.transform(data_numpy)
    
    # Create a DataFrame for Plotly
    n_points = embedding.shape[0]
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'time': range(n_points),
        'step': range(n_points)
    })
    
    # Create figure using same structure as embedded_plot
    fig = go.Figure()
    
    # Add line for trajectory
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='lines',
        line=dict(
            color='rgba(100,100,100,0.5)',
            width=1
        ),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            color=df['time'],
            colorscale='Viridis',
            size=6,
            opacity=0.8,
            showscale=True,
            colorbar=dict(
                title='Time'
            )
        ),
        text=[f"Step: {i}" for i in range(n_points)],
        hoverinfo='text',
        showlegend=False
    ))
    
    # Add start point (first step)
    fig.add_trace(go.Scatter(
        x=[embedding[0, 0]],
        y=[embedding[0, 1]],
        mode='markers',
        marker=dict(
            color='green',
            size=12,
            line=dict(color='black', width=1)
        ),
        name='Start',
        showlegend=True
    ))
    
    # Add end point (last step)
    fig.add_trace(go.Scatter(
        x=[embedding[-1, 0]],
        y=[embedding[-1, 1]],
        mode='markers',
        marker=dict(
            color='red',
            size=12,
            line=dict(color='black', width=1)
        ),
        name='End',
        showlegend=True
    ))
    
    # Add arrows to show direction
    for i in range(0, n_points-1, max(1, n_points//10)):  # Add arrows at several points
        fig.add_annotation(
            x=embedding[i+1, 0],
            y=embedding[i+1, 1],
            ax=embedding[i, 0],
            ay=embedding[i, 1],
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='black'
        )
    
    # Update layout
    fig.update_layout(
        title=f"Trial ID: {trial_id}<br>Label: '{label}'",
        showlegend=True,
        width=800,
        height=600,
        xaxis=dict(
            title=None,
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title=None,
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        hovermode='closest'
    )
    
    return fig


def create_side_by_side_plot(embedded_data, compressed_tokens, label, trial_id):
    """Create a side-by-side comparison of embedded and compressed data."""
    #print(f"Creating side-by-side plot for trial {trial_id}...")
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Embedded Data", "Compressed Tokens"],
        horizontal_spacing=0.1
    )
    
    # Process embedded data - fit UMAP specific to this trial
    embedded_umap = get_umap_model()

    # If embedded_data has shape [T, C, H, W]
    # We need to reshape to [T, C*H*W] for UMAP
    embedded_numpy = embedded_data.detach().cpu().numpy()

    # Reshape if needed
    if len(embedded_numpy.shape) > 2:
        T = embedded_numpy.shape[0]
        # Flatten all dimensions except the first (time)
        embedded_numpy = embedded_numpy.reshape(T, -1)

    #print(f"UMAP input shape: {embedded_numpy.shape}")  # Debug info

    # Now fit UMAP with correctly shaped data
    embedded_umap.fit(embedded_numpy)
    embedded_embedding = embedded_umap.transform(embedded_numpy)
    n_points_embedded = embedded_embedding.shape[0]
    
    # Process compressed tokens - fit UMAP specific to this trial
    compressed_umap = get_umap_model()
    compressed_numpy = compressed_tokens.detach().cpu().numpy()
    compressed_umap.fit(compressed_numpy)
    compressed_embedding = compressed_umap.transform(compressed_numpy)
    n_points_compressed = compressed_embedding.shape[0]
    
    # Add embedded data traces
    # Line for trajectory
    fig.add_trace(
        go.Scatter(
            x=embedded_embedding[:, 0],
            y=embedded_embedding[:, 1],
            mode='lines',
            line=dict(
                color='rgba(100,100,100,0.5)',
                width=1
            ),
            hoverinfo='none',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=embedded_embedding[:, 0],
            y=embedded_embedding[:, 1],
            mode='markers',
            marker=dict(
                color=np.arange(n_points_embedded),
                colorscale='Viridis',
                size=6,
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title='Time',
                    x=0.46
                )
            ),
            text=[f"Step: {i}" for i in range(n_points_embedded)],
            hoverinfo='text',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Start point
    fig.add_trace(
        go.Scatter(
            x=[embedded_embedding[0, 0]],
            y=[embedded_embedding[0, 1]],
            mode='markers',
            marker=dict(
                color='green',
                size=12,
                line=dict(color='black', width=1)
            ),
            name='Start',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # End point
    fig.add_trace(
        go.Scatter(
            x=[embedded_embedding[-1, 0]],
            y=[embedded_embedding[-1, 1]],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                line=dict(color='black', width=1)
            ),
            name='End',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add arrows for embedded
    for i in range(0, n_points_embedded-1, max(1, n_points_embedded//10)):
        fig.add_annotation(
            x=embedded_embedding[i+1, 0],
            y=embedded_embedding[i+1, 1],
            ax=embedded_embedding[i, 0],
            ay=embedded_embedding[i, 1],
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='black',
            row=1, col=1
        )
    
    # Add compressed data traces
    # Line for trajectory
    fig.add_trace(
        go.Scatter(
            x=compressed_embedding[:, 0],
            y=compressed_embedding[:, 1],
            mode='lines',
            line=dict(
                color='rgba(100,100,100,0.5)',
                width=1
            ),
            hoverinfo='none',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=compressed_embedding[:, 0],
            y=compressed_embedding[:, 1],
            mode='markers',
            marker=dict(
                color=np.arange(n_points_compressed),
                colorscale='Viridis',
                size=6,
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title='Time',
                    x=1.0
                )
            ),
            text=[f"Step: {i}" for i in range(n_points_compressed)],
            hoverinfo='text',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Start point
    fig.add_trace(
        go.Scatter(
            x=[compressed_embedding[0, 0]],
            y=[compressed_embedding[0, 1]],
            mode='markers',
            marker=dict(
                color='green',
                size=12,
                line=dict(color='black', width=1)
            ),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # End point
    fig.add_trace(
        go.Scatter(
            x=[compressed_embedding[-1, 0]],
            y=[compressed_embedding[-1, 1]],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                line=dict(color='black', width=1)
            ),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add arrows for compressed
    for i in range(0, n_points_compressed-1, max(1, n_points_compressed//10)):
        fig.add_annotation(
            x=compressed_embedding[i+1, 0],
            y=compressed_embedding[i+1, 1],
            ax=compressed_embedding[i, 0],
            ay=compressed_embedding[i, 1],
            xref='x2',
            yref='y2',
            axref='x2',
            ayref='y2',
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='black',
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=f"Trial ID: {trial_id}<br>Label: '{label}'",
        showlegend=True,
        width=1600,
        height=700,
        hovermode='closest'
    )
    
    # Remove axis labels and ticks
    fig.update_xaxes(
        title=None,
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )
    fig.update_yaxes(
        title=None,
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )
    
    return fig


def get_word_count(text):
    """Get number of words in text."""
    if not text:
        return 0
    return len(text.split())


def process_training_data(train_embedded_loader, train_compressed_loader, output_dir):
    """Process all training data and generate visualizations."""
    print("Generating visualizations for training data...")
    
    # Create a dictionary of compressed data by trial_id for easy lookup
    compressed_dict = {}
    if train_compressed_loader:
        for batch in train_compressed_loader:
            trial_id = batch['trial_id'][0]
            # Move tensors to CPU
            batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            compressed_dict[trial_id] = batch
    
    # Counter for progress report
    count = 0
    
    # Process each embedded sample
    for batch in tqdm(train_embedded_loader, desc="Processing trials"):
        try:
            # Extract data and ensure it's on CPU
            embedded_data = batch['vqvae_embeddings'][0].cpu()
            label = batch['label'][0]
            trial_id = batch['trial_id'][0]
            
            # Get word count for filename
            word_count = get_word_count(label)
            
            # Create filename
            filename_base = f"{trial_id}_{word_count}w"
            
            # Check if we have matching compressed data
            has_compressed = trial_id in compressed_dict
            
            # Generate visualizations
            if has_compressed:
                # Get compressed data
                compressed_batch = compressed_dict[trial_id]
                compressed_tokens = compressed_batch['compressed_tokens'][0]
                
                # Create side-by-side visualization
                fig = create_side_by_side_plot(
                    embedded_data, compressed_tokens, label, trial_id
                )
                
                # Save the figure
                output_path = os.path.join(output_dir, f"{filename_base}_comparison.html")
                fig.write_html(output_path)
            else:
                # Create embedded-only visualization
                fig = create_embedded_plot(embedded_data, label, trial_id)
                
                # Save the figure
                output_path = os.path.join(output_dir, f"{filename_base}_embedded.html")
                fig.write_html(output_path)
            
            count += 1
            
        except Exception as e:
            print(f"Error processing trial {trial_id}: {str(e)}")
            continue
    
    print(f"Visualization generation complete! Created {count} visualizations in {output_dir}")
    return count


def main():
    """Main function to run the visualization tool."""
    print("=== BCI Data Trajectory Visualization Tool ===")
    
    # Setup directories
    etl_dir, embedded_dir, compressed_dir, output_dir = setup_directories()
    
    # Load data
    (train_embedded_loader, test_embedded_loader, val_embedded_loader,
     train_compressed_loader, test_compressed_loader, val_compressed_loader,
     embedded_dataset, compressed_dataset) = load_data(
        embedded_dir, etl_dir, compressed_dir
    )
    
    # Process training data - now each trial gets its own UMAP
    count = process_training_data(
        train_embedded_loader, train_compressed_loader, output_dir
    )
    
    print(f"Created {count} visualization files")
    print(f"Output directory: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()