
import itertools
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import pandas as pd
import random
import time

class SpkPowArray:
        #
        # The most basic SpkPowerArray object is an 8x8 electrode array.
        # SpeechBCI packages the data in a 3D numpy array.
        #

    def __init__(self, session_id, block_id, trial_id, spkpowarray, start_chan, end_chan, array_dim=8):
        self.session_id = session_id # Typically a date
        self.block_id = block_id
        self.trial_id = trial_id
        self.idkey = f'{session_id}_{block_id}_{trial_id}'
        self.num_samples = spkpowarray.shape[0]
        self.spkpowarray = self.extract_area6v(spkpowarray, start_chan, end_chan, array_dim)
        
    def extract_area6v(self, spkpowarray, start_chan, end_chan, array_dim):
        return np.reshape(spkpowarray[:,start_chan:end_chan], (spkpowarray.shape[0], array_dim, array_dim))
        
    
    def implot(self, time_start, time_end, time_step=1, interval=200):
        
            # Create a figure and axes
        fig, ax = plt.subplots()

            # Initialize the plot
        im = ax.imshow(self.spkpowarray[0], cmap='hot', origin='upper')
        cbar = plt.colorbar(im)
        ax.set_title(self.idkey)

            # Animation update function
        def update(itx):
            im.set_data(self.spkpowarray[itx])
            return [im]

        ani = FuncAnimation(fig,
                            update,
                            frames=np.arange(time_start, time_end, time_step),
                            interval=interval,
                            repeat=True,
                            blit=True)
        plt.show()
        
    def tsplot(self, ix, iy):
            # Create the plot
        x = range(self.num_samples)
        y = self.spkpowarray[:,ix,iy]
        plt.plot(x, y)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Time Series For: [' + str(ix) + ', ' + str(iy) + ']')
        plt.show()
