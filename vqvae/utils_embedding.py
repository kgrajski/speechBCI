""" """

#
# 17March2025 - actively working.
# Sequence: etl.py -> main_vqvae3D.py (training) -> main_vqvae3D.py (encoding) -> main_mmllm.py
#

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import plotly.express as px
from torchvision.utils import make_grid


def plot_embeddings_ts(embeddings, idkey, subdir):
    """
    Plots the time series of embeddings for a given idkey.

    Args:
        embeddings (torch.Tensor): The embeddings to plot.
        idkey (int): The idkey for the embeddings.
        subdir (str): The subdirectory to save the plot.
    """
    # Convert embeddings to a list of indices
    indices = torch.argmax(embeddings, dim=1).cpu().numpy()

    # Create a time series plot of the indices
    import plotly.express as px

    fig = px.line(
        x=np.arange(len(indices)), y=indices, labels={"x": "Time", "y": "Index"}
    )
    fig.update_layout(title=f"Time Series of Indices for idkey {idkey}")

    # Save the plot to an HTML file
    plot_filename = os.path.join(subdir, f"{idkey}_plot.html")
    fig.write_html(plot_filename)

    # Function to embed the study data


def embed_studydata(model, study_dataset, device, embed_dir):

    # Set the model to eval mode.
    model.eval()
    torch.no_grad()

    # Create the subdirectory for train and test embeddings if it doesn't exist.
    subdir = os.path.join(embed_dir, "train")
    os.makedirs(subdir, exist_ok=True)
    subdir = os.path.join(embed_dir, "test")
    os.makedirs(subdir, exist_ok=True)

    # Get the set of unique sample idkeys in the study_dataset
    sample_idkeys = set(study_dataset.sample_idkey)

    # Set up to identify the longest sample series.  This will
    # help determine the context window for the upcoming MM-LLM.
    max_series_len = 0

    # Set up to generate statistics on the embeddings.
    #
    vq_codes = []
    # For each sample idkey, embed the samples, add positional encoding,
    # and save the embeddings.
    for idkey in tqdm(sample_idkeys, desc="Processing idkeys"):
        indices = [
            i
            for i in range(len(study_dataset))
            if study_dataset.sample_idkey[i] == idkey
        ]
        indices = sorted(indices)  # Ensure the indices are in order
        # Get the data for the sample idkey
        subset = torch.utils.data.Subset(study_dataset, indices)
        dataloader = DataLoader(subset, batch_size=len(indices), shuffle=False)

        data = next(iter(dataloader)).to(device)
        data = data.to(device)
        z = model._encoder(data)
        z = model._pre_vq(z)
        _, quantized, _, embeddings = model._vq_vae(z)

        # Store the embeddings for the sample idkey.  For future stats.
        vq_codes.append([torch.argmax(embeddings, dim=1).cpu().numpy()])

        # Determine the subdirectory based on val_flag
        if study_dataset.val_flag[indices[0]]:
            subdir = os.path.join(embed_dir, "test")
        else:
            subdir = os.path.join(embed_dir, "train")
        filename = os.path.join(subdir, f"{idkey}.pt")
        torch.save(quantized.squeeze(), filename)

        # Plot the time series of embeddings
        plot_embeddings_ts(embeddings, idkey, subdir)

        if len(indices) > max_series_len:
            max_series_len = len(indices)
        # print(f"For idkey {idkey} of length {len(indices)} and data {data.shape}, saved embeddings {embeddings.shape}.")

    print(f"Maximum series length: {max_series_len}")

    # Flatten vq_codees and generate a histogram of the indices using plotly express and save
    vq_codes = np.concatenate(vq_codes, axis=None)
    fig = px.histogram(x=vq_codes, title="Histogram of VQ Indices")
    fig.update_layout(bargap=0.1)
    fig.write_html(os.path.join(embed_dir, "histogram.html"))
