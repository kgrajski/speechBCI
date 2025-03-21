"""
This module sets up and runs the main experiment for a 3D Vector Quantized Variational Autoencoder (VQVAE)
model on Speech BCI data. It includes the main function to initialize the experiment and run it.

Functions:
    main(): Main function to set up the experiment and run it.
"""

"""
    Reminder: To monitor GPU utilization, use the following command:
        nvidia-smi --id=0 --loop=30 --query --display=UTILIZATION

    Reminder: To view TensorBoard logs, start TensorBoard on the command line with:
        tensorboard --logdir="/home/ubuntu/speechBCI/data/competitionData/tensorboard/VQVAE_256_256"
    Then open a browser tab to http://localhost:6006/
"""

#
# 14March2025 - actively working.
# Sequence: etl.py -> main_vqvae3D.py (training) -> main_vqvae3D.py (encoding) -> main_mmllm.py
#

import sys

sys.path.append("./")

import gc
import numpy as np
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

from SpeechBCIDataSet_3D import SpeechBCIDataSet_3D
from Vqvae_Simple3D import VQVAE
from utils_vqvae import run_exp
from utils_embedding import embed_studydata


def main():
    """
    Main function to set up the experiment and run it.
    """
    script_name = "main_vqvae3D"
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

    exp_name = "VQVAE_256_256"

    etl_dir = "/home/ubuntu/speechBCI/data/competitionData/etl"
    embed_dir = "/home/ubuntu/speechBCI/data/competitionData/embeddings"

    model_dir = "/home/ubuntu/speechBCI/data/competitionData/models"
    model_dir = os.path.join(model_dir, exp_name)
    os.makedirs(model_dir, exist_ok=True)

    tensorboard_dir = "/home/ubuntu/speechBCI/data/competitionData/tensorboard"
    tensorboard_dir = os.path.join(tensorboard_dir, exp_name)
    os.makedirs(tensorboard_dir, exist_ok=True)

    num_epochs = 10
    embedding_dim = 256
    num_embeddings = 256

    encoder_depth = 8  # Recall convention: B,C,D,H,W; 8 is the depth of the encoder
    depth_step_size = 2  # The depth stride when making samples from the raw input data.

    learning_rate = 1e-3

    training = True
    encoding = True

    test_prop = 0.2
    train_prop = 1 - test_prop
    batch_size = 512

    #
    # Per Willett, et al. competition data, the last block in each session
    # should be used as the test set.  Here, we'll call that set the validation set
    # and split the remaining data into training and validation sets.
    # Note: in the official competition, there is an identified validation set.
    #
    torch.autograd.set_detect_anomaly(True)
    study_dataset = SpeechBCIDataSet_3D(etl_dir, encoder_depth, depth_step_size)
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

    model = VQVAE(embedding_dim, num_embeddings).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    if training:
        run_exp(
            exp_name,
            model,
            train_dl,
            test_dl,
            val_dl,
            optimizer,
            device,
            num_epochs=num_epochs,
            model_dir=model_dir,
            show_plots=True,
            tensorboard_dir=tensorboard_dir,
        )

    if encoding:
        #
        # Use the trained model to generate embeddings for the train, test, and validation sets
        #  Print the model as a refresh and sanity check.
        #
        model.load_state_dict(
            torch.load(os.path.join(model_dir, exp_name + "_final" + ".pt"))
        )
        model.eval()
        print(model)
        embed_studydata(model, study_dataset, device, embed_dir)

    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")


if __name__ == "__main__":
    main()
