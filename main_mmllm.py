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
        tensorboard --logdir="/home/ubuntu/speechBCI/data/competitionData/tensorboard/VQVAE_Simple_3D"
    Then open a browser tab to http://localhost:6006/
"""

import sys
sys.path.append("./")

import gc
import numpy as np
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

from SpeechBCIDataSet_Embedded import SpeechBCIDataSet_Embedded
from utils_mmllm import run_exp

def main():
    """
    Main function to set up the experiment and run it.
    """
    script_name = "main_mmllm"
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
    
    exp_name = "MM_LLM"
    
    etl_dir = "/home/ubuntu/speechBCI/data/competitionData/etl"
    embed_dir = "/home/ubuntu/speechBCI/data/competitionData/embeddings"
    model_dir = "/home/ubuntu/speechBCI/data/competitionData/models"
    model_dir = os.path.join(model_dir, exp_name)
    tensorboard_dir = "/home/ubuntu/speechBCI/data/competitionData/tensorboard"
    tensorboard_dir = os.path.join(tensorboard_dir, exp_name)
    
    num_epochs = 100
    learning_rate = 1e-3
    training = False
    
    test_prop = 0.2
    train_prop = 1 - test_prop
    batch_size = 256
    
    #
    # Per Willett, et al. competition data, the last block in each session
    # should be used as the test set.  Here, we'll call that set the validation set
    # and split the remaining data into training and validation sets.
    # Note: in the official competition, there is a distinct validation (holdout) set.
    #
    #torch.autograd.set_detect_anomaly(True)
    study_dataset = SpeechBCIDataSet_Embedded(etl_dir, embed_dir)
    
    #
    # Recall that we are using competition data.  That study defines the last block in
    # each session as the test set.  And in such case the validation set is the data
    # that was withheld. At the risk of short changing the training set here, we'll
    # use the last block in each session as the withheld validation set.  The remaining
    # data we'll split into the traditional training and test set.
    # Consequently, it makes sense to have a quick 
    #
    # Generate some statistics on the words in the training and testing sets.
    #
    study_dataset._train_test_label_compare()
    
    # Now subset the study data as described above.
    train_test_indices = [i for i in range(len(study_dataset.val_flag)) if study_dataset.val_flag[i] is False]
    val_indices = [i for i in range(len(study_dataset.val_flag)) if study_dataset.val_flag[i] is True]
    train_test_dataset = Subset(study_dataset, train_test_indices)
    train_dataset, test_dataset = random_split(train_test_dataset, [train_prop, test_prop])
    val_dataset = Subset(study_dataset, val_indices)
    
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = None
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    
    if training:
        run_exp(exp_name, model, train_dl, test_dl, val_dl, optimizer, device, num_epochs=num_epochs,
                model_dir=model_dir, show_plots=True, tensorboard_dir=tensorboard_dir)

    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")
            
if __name__ == "__main__":
    main()