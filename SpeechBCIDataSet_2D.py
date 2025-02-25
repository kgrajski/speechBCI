import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import Dataset

from SpeechBCI import ElectrodeArray
from Sentence import Sentence

class SpeechBCIDataSet_2D(Dataset):
    """
    PyTorch custom Dataset tuned for Speech BCI Array Recordings.
    
    We treat each 2D array as an independent sample.
    
    And in this simple starting point, we'll faltten the 2D to 1D.
    
    That means that one input file will generate multiple samples.
    To do this, we spoof the classic sample, label pair in gen_dataset.
    
    Note:
        Adhere to the convention of NTCHW.

    Args:
        Dataset (_type_): _description_
    """
    
    def __init__(self, etl_dir, transform=None, target_transform=None):
        self.samples = self.gen_dataset(etl_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def gen_dataset(self, etl_dir):
            # We can make life simple by just pointing to a directory.
            # The directory will contain the ETL files.
            # We'll use the ETL file naming convention to generate the samples.
            # Based on the etl file naming convention, get basefile and variable names.
            # Get the unique idkeys and variable names and generate a list of paths.
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
        
            # Assemble the TCHW array (for later conversion to tensors).
            # Note the objects being read are Tx1x(HxW) arrays after reshaping.
        samples = []
        for idkey in idkeys:
                # Create a list of Ti x H x W arrays.
            working_array = []
            for var_name in var_names:
                fname = os.path.join(etl_dir + os.sep + idkey + '_' + var_name + '.csv')
                x = ElectrodeArray()
                x.load(fname)
                #print(f"Loaded {fname} shape {x.xt.shape} reshaped {x.xt.reshape(-1, x.num_rows, x.num_cols).shape}.")
                working_array.append(x.xt.reshape(-1, x.num_rows, x.num_cols))
                
                # Stack the arrays to create a Ti x C x H x W array.
            working_array = np.stack(working_array, axis=1)
            #print(f"working array shape = {working_array.shape}.")
            
                # Create a list of Tj x C x H x W arrays.
            samples.append(working_array)
            #print(f"samples shape {len(samples)}.")
            #print([s.shape for s in samples])
            
            # Stack the arrays to create the final T x C x H x W array,
            # where T is the sum of the Ti's.
        samples = np.concatenate(samples, axis=0)
 
        return samples
    
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        return x