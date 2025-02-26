
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

from VQVAE import VQVAE
import vutils

#
# Get the ECoGDataSet class defs.
#
from SpeechBCIDataSet_2D import SpeechBCIDataSet_2D

#
# Local Code
# Eventually move to a helper function file.
#
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
        print(f"Epoch {iter+1}\n-------------------------------")
        train(train_dl, model, optimizer, writer, device)
        loss, _, _ = test(test_dl, model, writer, device)
    
            #
            # Validate the model
            #
    print('Validation Set')
    loss, _, _ = test(val_dl, model, writer, device)
    print("##### Done Exp =", exp_name, "\n\n")

#
# Adapted from: https://github.com/Vrushank264/VQVAE-PyTorch/blob/main/train.py
#
def train(loader, model, opt, writer, device, beta = 1.0, steps = 0):
    
    loop = tqdm(loader, leave = True, position = 0)
    model.train()
    for imgs in loop:
        
        imgs = imgs.to(device)
        opt.zero_grad()
        z_e_x, z_q_x, x_tilde = model(imgs)
        
        recon_loss = fun.mse_loss(x_tilde, imgs)
        vq_loss = fun.mse_loss(z_q_x, z_e_x.detach())
        commitment_loss = fun.mse_loss(z_e_x, z_q_x.detach())
        
        loss = recon_loss + vq_loss + beta * commitment_loss
        loss.backward()
        
        writer.add_scalar("loss/train/reconstruction", recon_loss.item(), steps)
        writer.add_scalar("loss/train/quantization", vq_loss.item(), steps)
        writer.add_scalar("loss/train/commitment", commitment_loss.item(), steps)
        
        opt.step()
        steps += 1
        
def test(loader, model, writer, device, steps = 0):
    
    loop = tqdm(loader, leave = True, position = 0)
    model.eval()
    with torch.no_grad():
        recon_loss, vq_loss, commitment_loss = 0.0, 0.0, 0.0
        for imgs, _ in loop:
            
            imgs = imgs.to(device)
            z_e_x, z_q_x, x_tilde = model(imgs)
            recon_loss += fun.mse_loss(x_tilde, imgs)
            vq_loss += fun.mse_loss(z_q_x, z_e_x)
            commitment_loss += fun.mse_loss(z_e_x, z_q_x)
        
        recon_loss /= len(loader)
        vq_loss /= len(loader)
        commitment_loss /= len(loader)
        
    writer.add_scalar("loss/test/reconstruction", recon_loss.item(), steps)
    writer.add_scalar("loss/test/quantization", vq_loss.item(), steps)
    writer.add_scalar("loss/test/commitment", commitment_loss.item(), steps)
    
    return recon_loss.item(), vq_loss.item(), commitment_loss.item()

def generate(model, imgs, device = torch.device('cuda')):
    
    model.eval()
    with torch.no_grad():
        
        imgs = imgs.to(device)
        _, _, x_tilde = model(imgs)
    
    return x_tilde
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
    batch_size = 32

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
    in_channels = 2 # Channels in the traditional sense of channels such as RGB in image. hard-coded for now.
    out_channels =  64 # Dimensionality of the latent space.
    learning_rate = 1e-3

    model = VQVAE(in_channels, out_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
    
    