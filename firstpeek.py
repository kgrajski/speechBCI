
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

from SpeechBCI import ElectrodeArray

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
block_id = int(mat_data['blockIdx'][0][0])
trial_id = 0
start_chan = 0
end_chan = 128

if 1:
    desc = 'spikePow'
    spkpow = ElectrodeArray(desc, session_id, block_id, trial_id, mat_data[desc][0,trial_id], start_chan, end_chan)
    
    fig = spkpow.implot(0, spkpow.num_samples, 1)
    fig.write_html(f'./figs/competitionData/train/{spkpow.idkey}_implot.html', auto_open=False)
    fig.show()

    addl_text='Superior'
    fig = spkpow.tsplot(0, 8, addl_text=addl_text)
    fig.write_html(f'./figs/competitionData/train/{spkpow.idkey}_{addl_text}_tsplot.html', auto_open=False)
    fig.show()
    
    addl_text='Ventral'
    fig = spkpow.tsplot(8, 16, addl_text=addl_text)
    fig.write_html(f'./figs/competitionData/train/{spkpow.idkey}_{addl_text}_tsplot.html', auto_open=False)
    fig.show()

if 1:
    desc = 'tx1'
    tx1 = ElectrodeArray(desc, session_id, block_id, trial_id, mat_data[desc][0,trial_id], start_chan, end_chan)
    
    fig = tx1.implot(0, tx1.num_samples, 1)
    fig.write_html(f'./figs/competitionData/train/{tx1.idkey}_implot.html', auto_open=False)
    fig.show()

    addl_text='Superior'
    fig = tx1.tsplot(0, 8, addl_text=addl_text)
    fig.write_html(f'./figs/competitionData/train/{tx1.idkey}_{addl_text}_tsplot.html', auto_open=False)
    fig.show()
    
    addl_text='Ventral'
    fig = tx1.tsplot(8, 16, addl_text=addl_text)
    fig.write_html(f'./figs/competitionData/train/{tx1.idkey}_{addl_text}_tsplot.html', auto_open=False)
    fig.show()
