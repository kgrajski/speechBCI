"""
This module defines the SpeechBCIDataSet_3D class, a custom PyTorch Dataset
designed for handling Speech BCI Array Recordings. Each trial forms a time
series of 2D "images".

Classes:
    SpeechBCIDataSet_3D: A custom PyTorch Dataset for Speech BCI Array Recordings.

Usage example:
    dataset = SpeechBCIDataSet_3D(etl_dir="/path/to/etl")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in dataloader:
        # process data
"""

import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import Dataset

from SpeechBCI import ElectrodeArray
from Sentence import Sentence

class SpeechBCIDataSet_3D(Dataset):
    """
    PyTorch custom Dataset tuned for Speech BCI Array Recordings.
    
    We treat each trial as a 3D array: time series of 2D images.
    
    Note:
        Adhere to the convention of NTCHW.

    Args:
        etl_dir (str): Directory containing the ETL files.
        transform (callable, optional): Optional transform to be applied on a sample.
        target_transform (callable, optional): Optional transform to be applied on the target.
    """
    
    def __init__(self, etl_dir, kernel_size, transform=None, target_transform=None):
            # The val_flag will have same length as samples, with True indicating validation sample.
        self.samples, self.val_flag = self.gen_dataset(etl_dir, kernel_size)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def gen_dataset(self, etl_dir, kernel_size, kernel_multiplier=2):
        """
        Generate the dataset by reading ETL files from the specified directory.

        Args:
            etl_dir (str): Directory containing the ETL files.

        Returns:
            np.ndarray: Array of samples.
        """
        basefile_names = [f.split('.')[0] for f in os.listdir(etl_dir) if f.endswith('.csv')]
        idkeys = []
        var_names = []
        for basefile in basefile_names:
            parts = basefile.split('_')
            if not (parts[-1] == 'sentenceText'):
                idkeys.append('_'.join(parts[0:-3]))
                var_names.append('_'.join(parts[-3:]))
        idkeys = list(set(idkeys))
        var_names = set(var_names)
        print(f"Found {len(idkeys)} unique idkeys and {len(var_names)} unique variable names {var_names}.")
        var_names = list(var_names)
        
        samples = []
        val_flag = []
        for idkey in idkeys:
            working_array = []
            for var_name in var_names:
                fname = os.path.join(etl_dir + os.sep + idkey + '_' + var_name + '.csv')
                x = ElectrodeArray()
                x.load(fname) # (D or T) x H x W
                working_array.append(x.xt.reshape(1, -1, x.num_rows, x.num_cols))
            working_array = np.concatenate(working_array, axis=0) #@ C x T x H x W
            if int(x.block_id) == x.max_block_id:
                tmp_val_flag = True
            else:
                tmp_val_flag = False
            for islice in range(working_array.shape[1] - kernel_multiplier * kernel_size + 1):
                tmp = working_array[:, islice:islice+kernel_multiplier*kernel_size, :, :]
                samples.append(tmp) # CxTxHxW
                val_flag.append(tmp_val_flag)

        print(f"Generated {len(samples)} samples including {sum(val_flag)} for validation.")
        return samples, val_flag
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: The sample at the specified index.
        """
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        return x