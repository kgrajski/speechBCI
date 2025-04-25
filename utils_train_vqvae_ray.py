import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from SpeechBCIDataSet_3D import SpeechBCIDataSet_3D
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ray import tune
from Vqvae_Simple3D import VQVAE


def setup_data_loaders(
    data_dir,
    batch_size,
    encoder_depth,
    depth_step_size,
    train_prop=0.8,
    test_prop=0.2,
):
    """
    Sets up the dataset and data loaders.

    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Batch size for the data loaders.
        encoder_depth (int): Encoder depth for the dataset.
        depth_step_size (int): Depth step size for the dataset.

    Returns:
        tuple: train_dl, test_dl, val_dl (DataLoaders for training, testing, and validation).
    """

    etl_dir = os.path.join(data_dir, "etl", "6v_all")
    study_dataset = SpeechBCIDataSet_3D(etl_dir, encoder_depth, depth_step_size)

    train_test_indices = [
        i for i in range(len(study_dataset.val_flag)) if not study_dataset.val_flag[i]
    ]
    val_indices = [
        i for i in range(len(study_dataset.val_flag)) if study_dataset.val_flag[i]
    ]

    train_test_dataset = Subset(study_dataset, train_test_indices)
    train_dataset, test_dataset = random_split(
        train_test_dataset, [train_prop, test_prop]
    )
    val_dataset = Subset(study_dataset, val_indices)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl, val_dl


def train_vqvae(
    config,
    model_dir,
    tensorboard_dir,
    device,
    data_dir,
    batch_size,
    encoder_depth,
    depth_step_size,
):
    """
    Train a VQ-VAE model with the given configuration.

    Args:
        config (dict): Configuration dictionary containing hyperparameters.
        model_dir (str): Directory to save the model.
        tensorboard_dir (str): Directory to save TensorBoard logs.
        device (str): Device to run the model on ("cpu" or "cuda").
        data_dir (str): Directory containing the dataset.
        batch_size (int): Batch size for data loaders.
        encoder_depth (int): Encoder depth for the dataset.
        depth_step_size (int): Depth step size for the dataset.
    """
    # Setup data loaders locally in each trial
    train_dl, test_dl, val_dl = setup_data_loaders(
        data_dir,
        batch_size,
        encoder_depth,
        depth_step_size,
    )

    # Model setup
    model = VQVAE(
        config["num_ecog_input_channels"],
        config["num_encoder_out_channels"],
        config["embedding_dim"],
        config["num_embeddings"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train the model
    run_exp(
        exp_name=f"VQ_VAE_{config['embedding_dim']}_{config['num_embeddings']}",
        model=model,
        train_dl=train_dl,
        test_dl=test_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        device=device,
        num_epochs=config["num_epochs"],
        model_dir=model_dir,
        tensorboard_dir=tensorboard_dir,
        num_embeddings=config["num_embeddings"],
    )


def train(
    loader,
    model,
    optimizer,
    device,
    num_embeddings,
):
    """
    Trains the model for one epoch with progress displayed using tqdm.

    Args:
        loader (DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        device (str): Device to run the model on ("cpu" or "cuda").
        num_embeddings (int): Number of embedding vectors used in the model.

    Returns:
        tuple: Average reconstruction loss, VQ loss, and normalized perplexity.
    """
    loop = tqdm(
        loader,
        desc="Training",
        leave=True,
        position=0,
        disable=True,
    )
    data_recon_avg = []
    vq_loss_avg = []
    perplexity_avg = []
    model.train()
    for data in loop:
        data = data.to(device)
        optimizer.zero_grad()
        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / 255.0
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()

        data_recon_avg.append(recon_error.item())
        vq_loss_avg.append(vq_loss.item())
        perplexity_avg.append(
            perplexity.item() / num_embeddings
        )  # Normalize perplexity

        # Update tqdm progress bar with current losses
        loop.set_postfix(
            recon_loss=recon_error.item(),
            vq_loss=vq_loss.item(),
            perplexity=(perplexity.item() / num_embeddings),
        )

    data_recon_avg = np.mean(data_recon_avg)
    vq_loss_avg = np.mean(vq_loss_avg)
    perplexity_avg = np.mean(perplexity_avg)

    return data_recon_avg, vq_loss_avg, perplexity_avg


def test(
    loader,
    model,
    device,
    num_embeddings,
):
    """
    Tests the model with progress displayed using tqdm.

    Args:
        loader (DataLoader): DataLoader for the test data.
        model (torch.nn.Module): The model to test.
        device (str): Device to run the model on ("cpu" or "cuda").
        num_embeddings (int): Number of embedding vectors used in the model.

    Returns:
        tuple: Average reconstruction loss, VQ loss, and normalized perplexity.
    """
    loop = tqdm(
        loader,
        desc="Testing",
        leave=True,
        position=0,
        disable=True,
    )
    data_recon_avg, vq_loss_avg, perplexity_avg = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for data in loop:
            data = data.to(device)
            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data) / 255.0

            data_recon_avg += recon_error.item()
            vq_loss_avg += vq_loss.item()
            perplexity_avg += perplexity.item() / num_embeddings  # Normalize perplexity

            # Update tqdm progress bar with current losses
            loop.set_postfix(
                recon_loss=recon_error.item(),
                vq_loss=vq_loss.item(),
                perplexity=(perplexity.item() / num_embeddings),
            )

    data_recon_avg /= len(loader)
    vq_loss_avg /= len(loader)
    perplexity_avg /= len(loader)

    return data_recon_avg, vq_loss_avg, perplexity_avg


def run_exp(
    exp_name,
    model,
    train_dl,
    test_dl,
    val_dl,
    optimizer,
    device,
    num_epochs=1,
    model_dir=None,
    tensorboard_dir=None,
    num_embeddings=None,
    show_plots=True,
):
    """
    Runs the experiment, including training, testing, validation, and logging with Ray Tune.

    Args:
        exp_name (str): Name of the experiment.
        model (torch.nn.Module): The model to train and evaluate.
        train_dl (DataLoader): DataLoader for the training data.
        test_dl (DataLoader): DataLoader for the test data.
        val_dl (DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        device (str): Device to run the model on ("cpu" or "cuda").
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 1.
        model_dir (str, optional): Directory to save the model. Defaults to None.
        tensorboard_dir (str, optional): Directory to save TensorBoard logs. Defaults to None.
        num_embeddings (int, optional): Number of embedding vectors used in the model.
    """
    model_dir = os.path.join(model_dir, exp_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(tensorboard_dir, exp_name))
    print(f"Tensorboard directory: {tensorboard_dir} {exp_name}")
    print(f"Model directory: {model_dir}")

    print("##### Start Exp =", exp_name)
    # print(model)

    for iepoch in range(num_epochs):
        print(f"Epoch {iepoch+1}\n-------------------------------")
        train_data_recon, train_vq_loss, train_perplexity = train(
            train_dl,
            model,
            optimizer,
            device,
            num_embeddings,
        )
        writer.add_scalar("loss/train/reconstruction", train_data_recon, iepoch)
        writer.add_scalar("loss/train/quantization", train_vq_loss, iepoch)
        writer.add_scalar("loss/train/perplexity", train_perplexity, iepoch)
        print(
            f"Train Loss: {train_data_recon}",
            f"VQ Loss: {train_vq_loss}",
            f"Normalized Perplexity: {train_perplexity}",
        )

        test_data_recon, test_vq_loss, test_perplexity = test(
            test_dl,
            model,
            device,
            num_embeddings,
        )
        writer.add_scalar("loss/test/reconstruction", test_data_recon, iepoch)
        writer.add_scalar("loss/test/quantization", test_vq_loss, iepoch)
        writer.add_scalar("loss/test/perplexity", test_perplexity, iepoch)
        print(
            f"Test Loss: {test_data_recon}",
            f"VQ Loss: {test_vq_loss}",
            f"Normalized Perplexity: {test_perplexity}",
        )

        tune.report(
            {
                "reconstruction_loss_train": train_data_recon,
                "quantization_loss_train": train_vq_loss,
                "perplexity_train": train_perplexity,
                "reconstruction_loss_test": test_data_recon,
                "quantization_loss_test": test_vq_loss,
                "perplexity_test": test_perplexity,
            }
        )

    # Save the final model after the last epoch
    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(model_dir, exp_name + "_final.pt"),
        )
        print(f"Final model saved to {os.path.join(model_dir, exp_name + '_final.pt')}")

    if show_plots:
        proj = umap.UMAP(
            n_neighbors=3,
            min_dist=0.1,
            metric="cosine",
        ).fit_transform(model._vq_vae._embedding.weight.data.cpu())

        fig, ax = plt.subplots()
        ax.scatter(proj[:, 0], proj[:, 1])
        ax.set_title("Embedding Space Representation")
        writer.add_figure("Embedding Plot", fig, global_step=0)

        model.eval()
        valid_originals = next(iter(val_dl)).to(device)
        vq_output_eval = model._pre_vq(model._encoder(valid_originals))
        _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
        valid_reconstructions = model._decoder(model._post_vq(valid_quantize))

        if len(valid_originals.shape) == 4:
            img_grid = make_grid(
                valid_originals,
                nrow=16,
                scale_each=True,
            )
        else:
            img_grid = make_grid(
                valid_originals[1, :, :, :, :].squeeze(0),
                nrow=16,
                scale_each=True,
            )
        # writer.add_image("Originals", img_grid)

        if len(valid_reconstructions.shape) == 4:
            img_grid = make_grid(
                valid_reconstructions,
                nrow=16,
                scale_each=True,
            )
        else:
            img_grid = make_grid(
                valid_reconstructions[1, :, :, :, :].squeeze(0),
                nrow=16,
                scale_each=True,
            )
        # writer.add_image("Reconstructions", img_grid)

        writer.add_graph(model, valid_originals)

    writer.close()
