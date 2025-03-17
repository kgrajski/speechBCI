"""
"""

#
# 14March2025 - actively working.
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
from torchvision.utils import make_grid

def get_vqvae_codebook_average(model):
    """
    Calculate the average of all embedding vectors in the VQ-VAE codebook.
    
    This function extracts the codebook embeddings from a trained VQ-VAE model
    and computes their mean. This can be useful for padding or initialization
    purposes when using the codebook representations.
    
    Args:
        model (VQVAE): A trained VQ-VAE model instance
        
    Returns:
        torch.Tensor: The average embedding vector with shape [embedding_dim]
    """
    # Get the vector quantizer (either standard or EMA)
    vq = model._vq_vae
    
    # Extract the embedding weights (codebook vectors)
    # Shape: [num_embeddings, embedding_dim]
    codebook = vq._embedding.weight.data
    
    # Calculate the average across all codebook vectors
    # Shape: [embedding_dim]
    avg_vector = torch.mean(codebook, dim=0)
    
    return avg_vector

def train(loader, model, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        loader (DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        device (str): Device to run the model on ("cpu" or "cuda").

    Returns:
        tuple: Average reconstruction loss, VQ loss, and perplexity.
    """
    loop = tqdm(loader, leave=True, position=0)
    data_recon_avg, vq_loss_avg, perplexity_avg = 0, 0, 0
    model.train()
    for data in loop:
        data = data.to(device)
        optimizer.zero_grad()
        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / 255.0
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()
        
        data_recon_avg += recon_error
        vq_loss_avg += vq_loss
        perplexity_avg += perplexity
    
    data_recon_avg /= len(loader)
    vq_loss_avg /= len(loader)
    perplexity_avg /= len(loader)
    
    return data_recon_avg, vq_loss_avg, perplexity_avg
        
def test(loader, model, device):
    """
    Tests the model.

    Args:
        loader (DataLoader): DataLoader for the test data.
        model (torch.nn.Module): The model to test.
        device (str): Device to run the model on ("cpu" or "cuda").

    Returns:
        tuple: Average reconstruction loss, VQ loss, and perplexity.
    """
    loop = tqdm(loader, leave=True, position=0)
    data_recon_avg, vq_loss_avg, perplexity_avg = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for data in loop:
            data = data.to(device)
            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data) / 255.0

            data_recon_avg += recon_error
            vq_loss_avg += vq_loss
            perplexity_avg += perplexity
        
    data_recon_avg /= len(loader)
    vq_loss_avg /= len(loader)
    perplexity_avg /= len(loader)
        
    return data_recon_avg, vq_loss_avg, perplexity_avg

def run_exp(exp_name, model, train_dl, test_dl, val_dl, optimizer, device, num_epochs=1,
            model_dir=None, tensorboard_dir=None):
    
    """
    Runs the experiment, including training, testing, validation, and visualization.

    Args:
        exp_name (str): Name of the experiment.
        model (torch.nn.Module): The model to train and evaluate.
        train_dl (DataLoader): DataLoader for the training data.
        test_dl (DataLoader): DataLoader for the test data.
        val_dl (DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        device (str): Device to run the model on ("cpu" or "cuda").
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 1.
        training (bool, optional): Whether to train the model. If False, loads the model from model_dir. Defaults to True.
        model_dir (str, optional): Directory to save/load the model. Defaults to None.
        show_plots (bool, optional): Whether to generate and save plots. Defaults to True.
    """
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    print("##### Start Exp =", exp_name)
    print(model)
    print(f"Total parameters: {count_parameters(model)}")
    print(f"Trainable parameters: {count_trainable_parameters(model, True)}")
    
    for iepoch in range(num_epochs):
        print(f"Epoch {iepoch+1}\n-------------------------------")
        data_recon, vq_loss, perplexity = train(train_dl, model, optimizer, device)
        writer.add_scalar("loss/train/reconstruction", data_recon.item(), iepoch)
        writer.add_scalar("loss/train/quantization", vq_loss.item(), iepoch)
        writer.add_scalar("loss/train/perplexity", perplexity.item(), iepoch)
        print(f"Train Loss: {data_recon.item()}", f"VQ Loss: {vq_loss.item()}", f"Perplexity: {perplexity.item()}")
        
        data_recon, vq_loss, perplexity = test(test_dl, model, device)
        writer.add_scalar("loss/test/reconstruction", data_recon.item(), iepoch)
        writer.add_scalar("loss/test/quantization", vq_loss.item(), iepoch)
        writer.add_scalar("loss/test/perplexity", perplexity.item(), iepoch)
        print(f"Test Loss: {data_recon.item()}", f"VQ Loss: {vq_loss.item()}", f"Perplexity: {perplexity.item()}")
        
        if (model_dir is not None):
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, exp_name + "_" + str(iepoch) + ".pt"))
        
    data_recon, vq_loss, perplexity = test(val_dl, model, device)
    writer.add_scalar("loss/val/reconstruction", data_recon.item(), iepoch)
    writer.add_scalar("loss/val/quantization", vq_loss.item(), iepoch)
    writer.add_scalar("loss/val/perplexity", perplexity.item(), iepoch)
    print(f"Validation Loss: {data_recon.item()}", f"VQ Loss: {vq_loss.item()}", f"Perplexity: {perplexity.item()}")

    if model_dir is not None:
        torch.save(model.state_dict(), os.path.join(model_dir, exp_name + "_final" + ".pt"))

    writer.close()
    
