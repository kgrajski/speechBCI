"""
This module defines and runs experiments for a Vector Quantized Variational Autoencoder (VQVAE)
model on Speech BCI data. It includes functions for training, testing, and validating the model,
as well as visualizing the results using TensorBoard.

Functions:
    count_parameters(model): Returns the total number of parameters in the model.
    count_trainable_parameters(model): Returns the number of trainable parameters in the model.
    run_exp(exp_name, model, train_dl, test_dl, val_dl, optimizer, device, num_epochs=1,
            training=True, model_dir=None, show_plots=True): Runs the experiment, including training,
            testing, validation, and visualization.
    train(loader, model, optimizer, device): Trains the model for one epoch.
    test(loader, model, device): Tests the model.
    main(): Main function to set up the experiment and run it.

Usage example:
    Run the script from the command line to train and evaluate the VQVAE model:
    $ python dev_vqvae.py
"""

import sys
sys.path.append("./")

from collections.abc import Iterable
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torchvision.utils import make_grid

from tqdm import tqdm
import umap

from Vqvae_Classic2D import VQVAE
import vutils

from SpeechBCIDataSet_2D import SpeechBCIDataSet_2D

#
# Local Code
# Eventually move to a helper function file.
#

#
# Reminder: nvidia-smi --id=0 --loop=30 --query --display=UTILIZATION
#

#
# Reminder: To view, start TensorBoard on the command line with:
#   tensorboard --logdir=runs
# ...and open a browser tab to http://localhost:6006/

def count_parameters(model):
    """
    Returns the total number of parameters in the model.

    Args:
        model (torch.nn.Module): The model to count parameters for.

    Returns:
        int: Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """
    Returns the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model to count trainable parameters for.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_exp(exp_name, model, train_dl, test_dl, val_dl, optimizer, device, num_epochs=1,
            training=True, model_dir=None, show_plots=True):
    """
    Runs the experiment, including training, testing, validation, and visualization.

    Args:
        exp_name (str): Name of the experiment.
        model (torch.nn.Module): The model to train and evaluate.
        train_dl (DataLoader): DataLoader for the training data.
        test_dl (DataLoader): DataLoader for the test data.
        val_dl (DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        device (str): Device to run the model on ('cpu' or 'cuda').
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 1.
        training (bool, optional): Whether to train the model. If False, loads the model from model_dir. Defaults to True.
        model_dir (str, optional): Directory to save/load the model. Defaults to None.
        show_plots (bool, optional): Whether to generate and save plots. Defaults to True.
    """
    writer = SummaryWriter(os.path.join('runs' + os.sep + exp_name))
    if training:
        print("##### Start Exp =", exp_name)
        print(model)
        print(f"Total parameters: {count_parameters(model)}")
        print(f"Trainable parameters: {count_trainable_parameters(model)}")
        
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
            
        data_recon, vq_loss, perplexity = test(val_dl, model, device)
        writer.add_scalar("loss/val/reconstruction", data_recon.item(), iepoch)
        writer.add_scalar("loss/val/quantization", vq_loss.item(), iepoch)
        writer.add_scalar("loss/val/perplexity", perplexity.item(), iepoch)
        print(f"Validation Loss: {data_recon.item()}", f"VQ Loss: {vq_loss.item()}", f"Perplexity: {perplexity.item()}")

        if model_dir is not None:
            torch.save(model.state_dict(), os.path.join(model_dir, exp_name + ".pt"))
    
    else:
        model.load_state_dict(torch.load(os.path.join(model_dir, exp_name + ".pt")))
        
    if show_plots:
        proj = umap.UMAP(n_neighbors=3, min_dist=0.1,
                         metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())
        fig, ax = plt.subplots()
        ax.scatter(proj[:,0], proj[:,1])
        ax.set_title("Embedding Space Representation")
        writer.add_figure("Embedding Plot", fig, global_step=0)

        model.eval()
        valid_originals = next(iter(val_dl)).to(device)
        vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
        _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
        valid_reconstructions = model._decoder(valid_quantize)
        
        img_grid = make_grid(valid_originals, nrow=32, scale_each=True)
        writer.add_image('Originals', img_grid)
        
        img_grid = make_grid(valid_reconstructions, nrow=32, scale_each=True)
        writer.add_image('Reconstructions', img_grid)
        
        writer.add_graph(model, valid_originals)

    writer.close()

def train(loader, model, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        loader (DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        device (str): Device to run the model on ('cpu' or 'cuda').

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
        device (str): Device to run the model on ('cpu' or 'cuda').

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

def main():
    """
    Main function to set up the experiment and run it.
    """
    script_name = "dev_vqvae"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device=", device)
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    
    etl_dir = "/home/ubuntu/speechBCI/data/competitionData/etl"
    model_dir = "/home/ubuntu/speechBCI/data/competitionData/models"
    
    val_prop = 0.2
    test_prop = 0.2
    train_prop = 1 - val_prop - test_prop
    batch_size = 512
    
    study_dataset = SpeechBCIDataSet_2D(etl_dir)
    train_dataset, val_dataset, test_dataset = random_split(study_dataset, [train_prop, val_prop, test_prop])
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    exp_name = "VQVAE_2D"
    num_epochs = 100
    encoder_in_channels = 2
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 32
    num_embeddings = 64
    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-3
    training = True

    model = VQVAE(encoder_in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    
    run_exp(exp_name, model, train_dl, test_dl, val_dl, optimizer, device, num_epochs=num_epochs,
            training=training, model_dir=model_dir, show_plots=True)

    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")
            
if __name__ == '__main__':
    main()

