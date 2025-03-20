"""
This module sets up and runs the main experiment for a 3D Vector Quantized Variational Autoencoder (VQVAE)
model on Speech BCI data using Ray for hyperparameter tuning. It includes the main function to initialize the experiment and run it.

Functions:
    main(): Main function to set up the experiment and run it.

    Reminder: To monitor GPU utilization, use the following command:
        nvidia-smi --id=0 --loop=30 --query --display=UTILIZATION

    Reminder: To view TensorBoard logs, start TensorBoard on the command line with:
        tensorboard --logdir=runs
    Then open a browser tab to http://localhost:6006/
"""

#
# 14March2025 - actively working.  May need to be depcrated or at least carefully reviewed
# before applying since have redone VQVAE and other modules.  It is a good reference though.
#

import sys

sys.path.append("./")

import gc
import numpy as np
import os
import time
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import reduce
from operator import mul

from Vqvae_Simple3D import VQVAE
from utils_vqvae_ray import train_vqvae


def main():
    """
    Main function to set up the experiment and run it.
    """
    script_name = "main_vqvae3D_ray_rev1"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device=", device)
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    etl_dir = "/home/ubuntu/speechBCI/data/competitionData/etl"
    model_dir = "/home/ubuntu/speechBCI/data/competitionData/models"
    ray_dir = "/home/ubuntu/speechBCI/data/competitionData/ray_results"

    config = {
        "etl_dir": etl_dir,
        "model_dir": model_dir,
        "exp_name": "VQVAE_Simple_3D_1",
        "num_epochs": 5,
        "encoder_depth": tune.choice([32]),
        "encoder_in_channels": 2,
        "encoder_out_channels": tune.choice([64, 128]),
        "kernel_size": 4,
        "stride": 1,
        "padding": 1,
        "num_resid_layers": 2,
        "num_resid_channels": tune.choice([32]),
        "embedding_dim": tune.choice([64, 128]),
        "num_embeddings": tune.choice([128, 256]),
        "commitment_cost": 0.25,
        "decay": 0.99,
        "learning_rate": tune.choice([0.001]),
        "batch_size": 256,
        "test_prop": 0.2,
    }

    # Calculate num_samples as the product of the lengths of tune.choice elements
    num_samples = 8
    metric = "reconstruction_loss_train"
    mode = "min"

    scheduler = ASHAScheduler(max_t=100, grace_period=10, reduction_factor=2)

    analysis = tune.run(
        train_vqvae,
        resources_per_trial={"cpu": 4, "gpu": 0.99},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        metric=metric,
        mode=mode,
        storage_path=os.path.join(
            ray_dir, config["exp_name"]
        ),  # Combine ray_dir and exp_name
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")


if __name__ == "__main__":
    main()
