"""
This module defines the SpeechBCIDataSet_3D class, a custom PyTorch Dataset
designed for handling Speech BCI Array Recordings. Each trial forms a time
series of multi-channel 2D "images".

Classes:
    SpeechBCIDataSet_3D: A custom PyTorch Dataset for Speech BCI Array Recordings.

Usage example:
    dataset = SpeechBCIDataSet_3D(etl_dir="/path/to/etl", kernel_size=5)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in dataloader:
        # process data
"""

#
# 14March2025 - actively working.
# Sequence: etl.py -> main_vqvae3D.py (training) -> main_vqvae3D.py (encoding) -> main_mmllm.py
#

import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import Dataset

from etl.SpeechBCI import ElectrodeArray
from etl.Sentence import Sentence


class SpeechBCIDataSet_3D(Dataset):
    """
    PyTorch custom Dataset tuned for Speech BCI Array Recordings.

    We treat each trial as a 3D array: time series of 2D images.

    Note:
        Adhere to the convention of NTCHW.

    Args:
        etl_dir (str): Directory containing the ETL files.
        kernel_size (int): Size of the kernel to be used for slicing the time series.
        transform (callable, optional): Optional transform to be applied on a sample.
        target_transform (callable, optional): Optional transform to be applied on the target.
    """

    def __init__(
        self, etl_dir, depth, depth_step_size, transform=None, target_transform=None
    ):
        #
        # The val_flag will have same length as samples, with True indicating validation sample.
        # The sample_idkey will make it easy later to recombine samples to the original data.
        # The embed_index will be used to track the embedding index for each sample - after the
        # embedding is applied with a trained model.
        # The slice_id will be used to track the slice index for each sample - just to make sure
        # that when embedding is done original order is preserved.  [An excess of caution here.]
        #
        (
            self.samples,
            self.val_flag,
            self.sample_idkey,
            self.embed_index,
            self.slice_id,
        ) = self.gen_dataset(etl_dir, depth, depth_step_size)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def gen_dataset(self, etl_dir, depth, depth_step_size):
        """
        Generate the dataset by reading ETL files from the specified directory.

        Args:
            etl_dir (str): Directory containing the ETL files.
            kernel_size (int): Size of the kernel to be used for slicing the time series.
            kernel_multiplier (int, optional): Multiplier for the kernel size. Default is 2.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Array of samples.
                - list: List of boolean flags indicating validation samples.
        """
        basefile_names = sorted(
            [f.split(".")[0] for f in os.listdir(etl_dir) if f.endswith(".csv")]
        )
        # basefile_names = basefile_names[:100]
        idkeys = []
        var_names = []
        for basefile in basefile_names:
            parts = basefile.split("_")
            if not (parts[-1] == "sentenceText"):
                idkeys.append("_".join(parts[0:-3]))
                var_names.append("_".join(parts[-3:]))
        idkeys = sorted(list(set(idkeys)))
        var_names = sorted(list(set(var_names)))
        print(
            f"Found {len(idkeys)} unique idkeys and {len(var_names)} unique variable names {var_names}."
        )

        samples = []
        val_flag = []
        sample_idkey = []
        embed_index = []
        slice_id = []
        for idkey in idkeys:
            working_array = []
            for var_name in var_names:
                fname = os.path.join(etl_dir + os.sep + idkey + "_" + var_name + ".csv")
                x = ElectrodeArray()
                x.load(fname)  # (D or T) x H x W
                working_array.append(x.xt.reshape(1, -1, x.num_rows, x.num_cols))
            working_array = np.concatenate(working_array, axis=0)  # @ C x T x H x W
            if int(x.block_id) == x.max_block_id:
                tmp_val_flag = True
            else:
                tmp_val_flag = False

                #
                #   Generate samples from the working_array.  A sample consists of
                #   "depth" number of frames.
                #
            islice = 0
            for iframe_start in range(
                0, working_array.shape[1] - depth + 1, depth_step_size
            ):
                tmp = working_array[:, iframe_start : (iframe_start + depth) : 1, :, :]
                samples.append(tmp)  # CxTxHxW
                val_flag.append(tmp_val_flag)
                sample_idkey.append(idkey)
                embed_index.append(None)
                slice_id.append(islice)
                islice += 1
        samples = np.array(samples)

        print(
            f"Generated {len(samples)} samples including {sum(val_flag)} for validation."
        )
        return samples, val_flag, sample_idkey, embed_index, slice_id

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
