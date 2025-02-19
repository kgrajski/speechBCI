
import ipympl
import itertools
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import pandas as pd
import random
import scipy
from scipy.stats import multivariate_normal
import time

from SpeechBCI import SpkPowArray

mat_file_path = './data/competitionData/train/t12.2022.05.05.mat'

try:
    mat_data = scipy.io.loadmat(mat_file_path)
    print("MAT file loaded successfully.")
        # Access data within the loaded dictionary
    for key in mat_data:
        print(key, len(mat_data[key]), type(mat_data[key]))

    print(mat_data['spikePow'].shape)
    for icol in range(mat_data['spikePow'].shape[1]):
        print(mat_data['spikePow'][0,icol].shape)
except FileNotFoundError:
    print(f"Error: The file '{mat_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    
# Create a SpkPowArray object

session_id = 't12.2022.05.05'
block_id = int(mat_data['blockIdx'][0])
trial_id = 0
start_chan = 0
end_chan = 64
print("IN: ",session_id, block_id, trial_id, mat_data['spikePow'][0,trial_id].shape, type(mat_data['spikePow'][0,trial_id]))
spkpow = SpkPowArray('t12.2022.05.05', block_id, trial_id, mat_data['spikePow'][0,trial_id], start_chan, end_chan)
print(spkpow.idkey, spkpow.num_samples, spkpow.spkpowarray.shape)
print(spkpow.spkpowarray)
spkpow.implot(0, spkpow.num_samples, 1)

spkpow.tsplot(5,5)