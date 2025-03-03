"""
This module defines the ElectrodeArray class, which represents a time series of 2D arrays of channels
from an electrode array. It provides methods to extract data, plot the data as heatmaps or time series,
and save/load the data to/from a CSV file.

Classes:
    ElectrodeArray: A class to represent a time series of 2D arrays of channels from an electrode array.

Usage example:
    electrode_array = ElectrodeArray(description="example", session_id="session1", max_block_id=10, block_id=1,
                                     trial_id=1, xt=np.random.rand(100, 128), start_chan=0, end_chan=64)
    electrode_array.save("/path/to/output")
    electrode_array.load("/path/to/output/session1_1_1_example.csv")
    fig = electrode_array.implot(time_start=0, time_end=100)
    fig.show()
    fig = electrode_array.tsplot(start_row=0, end_row=8)
    fig.show()
"""

import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio

class ElectrodeArray:
    """
    A basic electrode array class: time series of 2D array of channels.
    It is the result obtained after doing data extraction from some
    given data source (e.g., Willett, et al. (2023) data).
    Goal: organize things so that they play nicely with PyTorch.

    The results for self.xt is that one can reference the data by 2D
    array coordinates where (0,0) is the anterior, superior electrode.
    Rows represent superior to inferior.
    Columns represent anterior to posterior.
    """
    def __init__(self, description='', session_id='', max_block_id=0, block_id=0, trial_id=0, xt=None,
                 start_chan=0, end_chan=0, num_rows=16, num_cols=8):
        self.session_id = session_id.replace('.', '_')
        self.block_id = block_id
        self.desc = description  # convention: use raw data variable name
        self.idkey = f'{self.session_id}_{block_id}_{trial_id}_{description}'  # unique identifier
        self.max_block_id = max_block_id
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_samples = xt.shape[0] if xt is not None else 0
        self.trial_id = trial_id
        self.xt = self.extract_area6v(xt, start_chan, end_chan)

    def extract_area6v(self, xt, start_chan, end_chan):
        """
        Extracts the data for Area 6v based on the channel mapping.
        """
        chan_to_electrode_map = [62, 51, 43, 35, 94, 87, 79, 78,
                                 60, 53, 41, 33, 95, 86, 77, 76,
                                 63, 54, 47, 44, 93, 84, 75, 74,
                                 58, 55, 48, 40, 92, 85, 73, 72,
                                 59, 45, 46, 38, 91, 82, 71, 70,
                                 61, 49, 42, 36, 90, 83, 69, 68,
                                 56, 52, 39, 34, 89, 81, 67, 66,
                                 57, 50, 37, 32, 88, 80, 65, 64,
                                 125, 126, 112, 103, 31, 28, 11, 8,
                                 123, 124, 110, 102, 29, 26, 9, 5,
                                 121, 122, 109, 101, 27, 19, 18, 4,
                                 119, 120, 108, 100, 25, 15, 12, 6,
                                 117, 118, 107, 99, 23, 13, 10, 3,
                                 115, 116, 106, 97, 21, 20, 7, 2,
                                 113, 114, 105, 98, 17, 24, 14, 0,
                                 127, 111, 104, 96, 30, 22, 16, 1]
        if xt is not None:
            k = end_chan - start_chan
            extracted_xt = np.zeros((self.num_samples, k))
            for iel in range(k):
                extracted_xt[:, iel] = xt[:, chan_to_electrode_map[start_chan + iel]]
        else:
            extracted_xt = None
            
        return extracted_xt

    def implot(self, time_start, time_end, time_step=1, interval=100):
        """
        Using plotly express, plot the electrode array (xt) as a
        time-series of 2D image heatmaps.
        Note: the data stored as 1D array, so reshape to 2D.
        """
        pio.renderers.default = "browser"
        
        # Data Integrity Check - reshaping can be a common source of errors
        num_frames = (time_end - time_start) // time_step
        expected_size = num_frames * self.num_rows * self.num_cols
        data = self.xt[time_start:time_end:time_step]
        if data.size != num_frames * self.num_rows * self.num_cols:
            raise ValueError(
                f"Data size mismatch: expected {expected_size}, "
                f"got {data.size}")
        
        data = data.reshape((-1, self.num_rows, self.num_cols))
        
        fig = px.imshow(data, animation_frame=0,
                        color_continuous_scale="Hot", origin='upper',
                        aspect='equal',
                        labels=dict(animation_frame="Time Step",
                                    x="X-axis", y="Y-axis",
                                    color=self.desc))
        
        fig.update_layout(
            title=f'Array: {self.idkey}',
            xaxis_title="X-axis",
            yaxis_title="Y-axis"
        )
        
        # Safely set animation speed
        if fig.layout.updatemenus and fig.layout.updatemenus[0].buttons:
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"][
                "duration"] = interval
        
        return fig

    def tsplot(self, start_row, end_row, interval=100):
        """
        Using plotly express, plot a specified set of electrode array
        x,y positions as time series using stack and facet features to
        create one plot per electrode.
        For each time point, the spatial data stored at each as 1D array.
        Indicate starting and ending spatial points as linear indices.
        Then reshape the data to 2D.
        start_row and end_row - use Python indexing convention.
        """
        pio.renderers.default = "browser"
    
        # Data Integrity Check
        expected_rows = end_row - start_row
        if not (0 < expected_rows <= self.num_rows):
            raise ValueError(
                f"Invalid row range: expected 1 to {self.num_rows}, "
                f"got {expected_rows}")
    
        istart = start_row * self.num_cols
        iend = end_row * self.num_cols
        df = pd.DataFrame(self.xt[:,istart:iend].reshape(
            (self.num_samples, expected_rows * self.num_cols)))

        df = df.stack().reset_index()
        df.columns = ['time', 'el', 'val']
    
        fig = px.line(df, x='time', y='val', facet_col='el',
                      facet_col_wrap=self.num_cols,
                      title=f'Time Series: {self.idkey}')
        fig.update_layout(xaxis_title="Time", yaxis_title=self.desc)
    
        # Safe Animation Duration Update
        if fig.layout.updatemenus and fig.layout.updatemenus[0].buttons:
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"][
                "duration"] = interval
    
        return fig

    def save(self, out_dir):
        """
        Create a file name and write a csv file.
        """
        fname = os.path.join(out_dir + os.sep + self.idkey + '.csv')
        try:
            with open(fname, 'w') as f:
                f.write(f"{self.block_id}\n")
                f.write(f"{self.desc}\n")
                f.write(f"{self.idkey}\n")
                f.write(f"{self.max_block_id}\n")
                f.write(f"{self.num_cols}\n")
                f.write(f"{self.num_rows}\n")
                f.write(f"{self.num_samples}\n")
                f.write(f"{self.session_id}\n")
                f.write(f"{self.trial_id}\n")
                if self.xt is not None and self.xt.size > 0:
                    np.savetxt(f, self.xt, fmt='%.8f', delimiter=',')
                else:
                    print(f"Warning: self.xt is empty for {self.idkey}, not saving data.")
        except Exception as e:
            print(f"Error saving data to {fname}: {e}")

    def load(self, fname):
        """
        Read a (completed) file name and load a csv file for ecog.
        """
        try:
            with open(fname, 'r') as f:
                self.block_id = f.readline().strip()
                self.desc = f.readline().strip()
                self.idkey = f.readline().strip()
                self.max_block_id = int(f.readline().strip())
                self.num_cols = int(f.readline().strip())
                self.num_rows = int(f.readline().strip())
                self.num_samples = int(f.readline().strip())
                self.session_id = f.readline().strip()
                self.trial_id = int(f.readline().strip())
                self.xt = np.loadtxt(f, delimiter=',')
        except Exception as e:
            print(f"Error loading data from {fname}: {e}")
