
import sys
sys.path.append("./")

import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

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

#
# MAIN
#
def main():

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
    batch_size = 60

        #
        # Directory containing the ETL data.
        #
    etl_data_dir = "/home/ubuntu/speechBCI/data/competitionData/etl"
    
        #
        # Make a study dataset
        #   Note: Choice of loss function could require one-hot encoding of the label vector.
        #
    study_dataset = SpeechBCIDataSet_2D(etl_data_dir)

        #
        # Make the train, validation, and test splits
        #
    #train_dataset, val_dataset, test_dataset = random_split(study_dataset, [train_prop, val_prop, test_prop])

        #
        # Setup DataLoaders
        #
    #train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        #
        # OPTIONAL: Quick sanity checks to make sure data read in correctlyl
        #
    if 0:
        
        print("** SUMMARY **", study_dataset.samples.groupby(['label'])['label'].count())
     
        ecog_tensor, label_tensor = train_dataset.__getitem__(0)
        print('Train ECoG shape', ecog_tensor.shape, label_tensor.shape)        

        print("TRAIN")
        train_batch = next(iter(train_dl))
        print('Batch shape',train_batch[0].shape)
        
        if 0:
            for batch_idx, (batch_data, batch_labels) in enumerate(val_batch):
                print(f"Batch index: {batch_idx}")
                print("Batch data shape:", batch_data.shape)
                print("Batch labels:", batch_labels)
                print("Batch labels shape:", batch_labels.shape)
    
        #
        #   Model Training
        #

            #
            #   Select Loss Function and Optimizer
            #
    num_epochs = 10
    lrn_rate = 0.001
    lrn_momentum = 0.9
    loss_fn = nn.CrossEntropyLoss()
        
            #
            #   Set up the model and run the experiment
            #

                #
                #   Dirt simple "flattened" model.
                #
    if 0:
        exp_name = "NN_Flat"
        input_dim = 32 * 64 * 32
        hidden_dim = 8
        output_dim = 6
        num_layers = 3
        model = NN_Flat(input_dim, hidden_dim, num_layers, output_dim).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lrn_rate, momentum=lrn_momentum)
        run_exp(exp_name, model, num_epochs, train_dl, test_dl, val_dl, loss_fn, optimizer, device)

        
                #
                #   Very simple Conv3D model.
                #
    if 0:
        exp_name = "NN_Conv3D_Simple"
        in_depth = 32
        in_rows = 64
        in_cols = 32
        fc_dim = 32
        output_dim = 6
        model = NN_Conv3D_Simple(in_depth, in_rows, in_cols, fc_dim, output_dim).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lrn_rate, momentum=lrn_momentum)
        run_exp(exp_name, model, num_epochs, train_dl, test_dl, val_dl, loss_fn, optimizer, device)

        #
        # Wrap-up
        #
    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")
            
#
# EXECUTE
#
if __name__ == '__main__':
    main()
    
    