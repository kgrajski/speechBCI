"""
This module defines utility functions for running experiments with a 3D Vector Quantized Variational Autoencoder (VQVAE)
model on Speech BCI data using Ray for hyperparameter tuning. It includes functions for counting parameters, training, testing, and running experiments.

Functions:
    count_parameters(model): Returns the total number of parameters in the model.
    count_trainable_parameters(model, show_details=False): Returns the number of trainable parameters in the model.
    train_vqvae(config, checkpoint_dir=None): Trains the VQVAE model with the given configuration.
    train(loader, model, optimizer, device): Trains the model for one epoch.
    test(loader, model, device): Tests the model.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
import umap
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from ray import tune

from SpeechBCIDataSet_3D import SpeechBCIDataSet_3D  # Ensure this import is correct
from Vqvae_Simple3D import VQVAE

#
# 14March2025 - not actively working.
# May need to be revised to work with redefined Vqvae_Simpl3D.py. But a decent reference.
# Sequence: etl.py -> main_vqvae3D.py (training) -> main_vqvae3D.py (encoding) -> main_mmllm.py
#


def count_parameters(model):
    """
    Returns the total number of parameters in the model.

    Args:
        model (torch.nn.Module): The model to count parameters for.

    Returns:
        int: Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model, show_details=False):
    """
    Returns the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model to count trainable parameters for.
        show_details (bool, optional): Whether to print details of each parameter. Defaults to False.

    Returns:
        int: Number of trainable parameters.
    """
    if show_details:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_vqvae(config, checkpoint_dir=None):
    """
    Trains the VQVAE model with the given configuration.

    Args:
        config (dict): Configuration dictionary containing hyperparameters.
        checkpoint_dir (str, optional): Directory to load checkpoint from. Defaults to None.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    etl_dir = config["etl_dir"]
    model_dir = config["model_dir"]
    exp_name = config["exp_name"]
    num_epochs = config["num_epochs"]
    encoder_depth = config["encoder_depth"]
    encoder_in_channels = config["encoder_in_channels"]
    encoder_out_channels = config["encoder_out_channels"]
    kernel_size = config["kernel_size"]
    stride = config["stride"]
    padding = config["padding"]
    num_resid_layers = config["num_resid_layers"]
    num_resid_channels = config["num_resid_channels"]
    embedding_dim = config["embedding_dim"]
    num_embeddings = config["num_embeddings"]
    commitment_cost = config["commitment_cost"]
    decay = config["decay"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    test_prop = config["test_prop"]
    train_prop = 1 - test_prop

    # torch.autograd.set_detect_anomaly(True)
    study_dataset = SpeechBCIDataSet_3D(etl_dir, encoder_depth)
    train_test_indices = [
        i
        for i in range(len(study_dataset.val_flag))
        if study_dataset.val_flag[i] is False
    ]
    val_indices = [
        i
        for i in range(len(study_dataset.val_flag))
        if study_dataset.val_flag[i] is True
    ]

    train_test_dataset = Subset(study_dataset, train_test_indices)
    train_dataset, test_dataset = random_split(
        train_test_dataset, [train_prop, test_prop]
    )
    val_dataset = Subset(study_dataset, val_indices)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = VQVAE(
        encoder_in_channels,
        encoder_out_channels,
        kernel_size,
        stride,
        padding,
        num_resid_layers,
        num_resid_channels,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    # writer = SummaryWriter(os.path.join("runs" + os.sep + exp_name))
    for iepoch in range(num_epochs):
        # print(f"Epoch {iepoch+1}\n-------------------------------")
        train_data_recon, train_vq_loss, train_perplexity = train(
            train_dl, model, optimizer, device
        )
        # writer.add_scalar("loss/train/reconstruction", data_recon.item(), iepoch)
        # writer.add_scalar("loss/train/quantization", vq_loss.item(), iepoch)
        # writer.add_scalar("loss/train/perplexity", perplexity.item(), iepoch)
        # print(f"Train Loss: {data_recon.item()}", f"VQ Loss: {vq_loss.item()}", f"Perplexity: {perplexity.item()}")

        test_data_recon, test_vq_loss, test_perplexity = test(test_dl, model, device)
        # writer.add_scalar("loss/test/reconstruction", data_recon.item(), iepoch)
        # writer.add_scalar("loss/test/quantization", vq_loss.item(), iepoch)
        # writer.add_scalar("loss/test/perplexity", perplexity.item(), iepoch)
        # print(f"Test Loss: {data_recon.item()}", f"VQ Loss: {vq_loss.item()}", f"Perplexity: {perplexity.item()}")

        tune.report(
            {
                "reconstruction_loss_train": train_data_recon.item(),
                "quantization_loss_train": train_vq_loss.item(),
                "perplexity_train": train_perplexity.item(),
                "reconstruction_loss_test": test_data_recon.item(),
                "quantization_loss_test": test_vq_loss.item(),
                "perplexity_test": test_perplexity.item(),
            }
        )

    # writer.close()


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
    data_recon_avg, vq_loss_avg, perplexity_avg = 0, 0, 0
    model.train()
    for data in loader:
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
    data_recon_avg, vq_loss_avg, perplexity_avg = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for data in loader:
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
