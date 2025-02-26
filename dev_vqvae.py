
import sys
sys.path.append("./")

from collections.abc import Iterable
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

from tqdm import tqdm

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

def run_exp(exp_name, model, train_dl, test_dl, val_dl, optimizer, device, num_epochs=10):
    
            #
            # Set up tensorboard
            # Default log_dir argument is "runs" - but it's good to be specific
            # torch.utils.tensorboard.SummaryWriter is imported above
            #
    writer = SummaryWriter(os.path.join('runs' + os.sep + exp_name))

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
    for iter in range(num_epochs):
        
            #
            # Train the model
            #
        print(f"Epoch {iter+1}\n-------------------------------")
        data_recon, vq_loss, perplexity = train(train_dl, model, optimizer, device)
        writer.add_scalar("loss/train/reconstruction", data_recon.item(), iter)
        writer.add_scalar("loss/train/quantization", vq_loss.item(), iter)
        writer.add_scalar("loss/train/perplexity", perplexity.item(), iter)
        print(f"Train Loss: {data_recon.item()}", f"VQ Loss: {vq_loss.item()}", f"Perplexity: {perplexity.item()}")
        
            #
            # Test the model
            #     
        data_recon, vq_loss, perplexity = test(test_dl, model, device)
        writer.add_scalar("loss/test/reconstruction", data_recon.item(), iter)
        writer.add_scalar("loss/test/quantization", vq_loss.item(), iter)
        writer.add_scalar("loss/test/perplexity", perplexity.item(), iter)
        print(f"Test Loss: {data_recon.item()}", f"VQ Loss: {vq_loss.item()}", f"Perplexity: {perplexity.item()}")
        
        #
        # Validate the model
        #
    data_recon, vq_loss, perplexity = test(val_dl, model, device)
    writer.add_scalar("loss/val/reconstruction", data_recon.item(), iter)
    writer.add_scalar("loss/val/quantization", vq_loss.item(), iter)
    writer.add_scalar("loss/val/perplexity", perplexity.item(), iter)

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

        # For reproducibility and fair comparisons
    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    
        #
        # Dataset Partitioning Parameters
        #
    val_prop = 0.2
    test_prop = 0.2
    train_prop = 1 - val_prop - test_prop
    batch_size = 1024

        #
        # Directory containing the ETL data.
        #
    etl_dir = "/home/ubuntu/speechBCI/data/competitionData/etl"
    
        #
        # Make a study dataset and then partition to train, val, and test.
        #
    study_dataset = SpeechBCIDataSet_2D(etl_dir)
    train_dataset, val_dataset, test_dataset = random_split(study_dataset, [train_prop, val_prop, test_prop])
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        #
        #   Create the model and run one or more experiments.
        #
    exp_name = "VQVAE_2D"
    encoder_in_channels = 2
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-3

    model = VQVAE(encoder_in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    
        #
        # Do the experiment.
        # Normally prefer to have a function call here, but for now, just do it.
        #
    run_exp(exp_name, model, train_dl, test_dl, val_dl, optimizer, device)

        #
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
    
    