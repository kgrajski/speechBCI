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

from Vqvae import VQVAE
import vutils

#
# Get the ECoGDataSet class defs.
#
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
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_exp(exp_name, model, train_dl, test_dl, val_dl, optimizer, device, num_epochs=1,
            training=True, model_dir=None, show_plots=True):
    
            #
            # Set up tensorboard
            # Default log_dir argument is "runs" - but it's good to be specific
            # torch.utils.tensorboard.SummaryWriter is imported above
            #
    writer = SummaryWriter(os.path.join('runs' + os.sep + exp_name))
    if training:

            #
            # Print some information about the model
            #
        print("##### Start Exp =", exp_name)
        print(model)
        print(f"Total parameters: {count_parameters(model)}")
        print(f"Trainable parameters: {count_trainable_parameters(model)}")
        
            #
            # Train the model
            #
        for iepoch in range(num_epochs):
            
                #
                # Train the model
                #
            print(f"Epoch {iepoch+1}\n-------------------------------")
            data_recon, vq_loss, perplexity = train(train_dl, model, optimizer, device)
            writer.add_scalar("loss/train/reconstruction", data_recon.item(), iepoch)
            writer.add_scalar("loss/train/quantization", vq_loss.item(), iepoch)
            writer.add_scalar("loss/train/perplexity", perplexity.item(), iepoch)
            print(f"Train Loss: {data_recon.item()}", f"VQ Loss: {vq_loss.item()}", f"Perplexity: {perplexity.item()}")
            
                #
                # Test the model
                #     
            data_recon, vq_loss, perplexity = test(test_dl, model, device)
            writer.add_scalar("loss/test/reconstruction", data_recon.item(), iepoch)
            writer.add_scalar("loss/test/quantization", vq_loss.item(), iepoch)
            writer.add_scalar("loss/test/perplexity", perplexity.item(), iepoch)
            print(f"Test Loss: {data_recon.item()}", f"VQ Loss: {vq_loss.item()}", f"Perplexity: {perplexity.item()}")
            
            #
            # Validate the model
            #
        data_recon, vq_loss, perplexity = test(val_dl, model, device)
        writer.add_scalar("loss/val/reconstruction", data_recon.item(), iepoch)
        writer.add_scalar("loss/val/quantization", vq_loss.item(), iepoch)
        writer.add_scalar("loss/val/perplexity", perplexity.item(), iepoch)
        print(f"Validation Loss: {data_recon.item()}", f"VQ Loss: {vq_loss.item()}", f"Perplexity: {perplexity.item()}")

        
            #
            # Save the model
            #
        if model_dir is not None:
            torch.save(model.state_dict(), os.path.join(model_dir, exp_name + ".pt"))
    
    else:
        model.load_state_dict(torch.load(os.path.join(model_dir, exp_name + ".pt")))
        
    if show_plots:

            #
            # Visualize the codebook.
            #
        proj = umap.UMAP(n_neighbors=3, min_dist=0.1,
                         metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())
        fig, ax = plt.subplots()
        ax.scatter(proj[:,0], proj[:,1])
        ax.set_title("Embedding Space Representation")
        writer.add_figure("Embedding Plot", fig, global_step=0)

            #
            # Visualize some original and reconstructed plots.
            #
        model.eval()
        valid_originals = next(iter(val_dl)).to(device)
        vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
        _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
        valid_reconstructions = model._decoder(valid_quantize)
        
        img_grid = make_grid(valid_originals, nrow=16)
        writer.add_image('Originals', img_grid)
        
        img_grid = make_grid(valid_reconstructions, nrow=16)
        writer.add_image('Reconstructions', img_grid)

        #
        # Close tensorboard
        #
    writer.close()

def train(loader, model, optimizer, device):
    
    loop = tqdm(loader, leave = True, position = 0)
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
    
    loop = tqdm(loader, leave = True, position = 0)
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

#
# Define main
#
def main():

    #
    # Reminder: nvidia-smi --id=0 --loop=30 --query --display=UTILIZATION
    #

    #
    # Reminder: To view, start TensorBoard on the command line with:
    #   tensorboard --logdir=runs
    # ...and open a browser tab to http://localhost:6006/


    #
    # Reminder: nvidia-smi --id=0 --loop=30 --query --display=UTILIZATION
    #

    #
    # Reminder: To view, start TensorBoard on the command line with:
    #   tensorboard --logdir=runs
    # ...and open a browser tab to http://localhost:6006/

        # Set script name for console log
    script_name = "dev_vqvae"
    
        # Start timer
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***\n")
    
        #
        # Determine device availability
        #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device=", device)
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

        # For reproducibility and fair comparisons
    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    
        #
        # Directory containing the ETL data.
        #
    etl_dir = "/home/ubuntu/speechBCI/data/competitionData/etl"
    model_dir = "/home/ubuntu/speechBCI/data/competitionData/models"
    
        #
        # Dataset Partitioning Parameters
        #
    val_prop = 0.2
    test_prop = 0.2
    train_prop = 1 - val_prop - test_prop
    batch_size = 256
    
        #
        # Make a study dataset and then partition to train, val, and test.
        #
    study_dataset = SpeechBCIDataSet_2D(etl_dir)
    train_dataset, val_dataset, test_dataset = random_split(study_dataset, [train_prop, val_prop, test_prop])
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        #
        # Create the model and run one or more experiments.
        #
    exp_name = "VQVAE_2D"
    num_epochs = 100
    encoder_in_channels = 2
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 256
    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-3
    training = True # If False, load the (same) model from the model_dir.

    model = VQVAE(encoder_in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    
        #
        # Do the experiment.
        #
    run_exp(exp_name, model, train_dl, test_dl, val_dl, optimizer, device, num_epochs=num_epochs,
            training=training, model_dir=model_dir, show_plots=True)

        #
        # Wrap-up
        #
    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")
            
#
# Run main
#
if __name__ == '__main__':
    main()

