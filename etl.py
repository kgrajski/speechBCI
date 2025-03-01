"""
This module performs ETL (Extract, Transform, Load) operations on Speech BCI data.
It processes raw data files, extracts relevant information, applies transformations
such as block-based Z-scoring, and saves the processed data for further analysis.

Functions:
    block_zscore(flist, etl_dir): Calculates global mean and std, then applies Z-scoring to the data.
    etl_blockZ(session_id, mat_data, etl_dir, stats_dir='', show_plots=False): Performs ETL operations on a session.
    main(): Main function to execute the ETL process on all identified .mat files.

Usage example:
    Run the script from the command line to process all .mat files in the specified directory:
    $ python etl.py
"""

import itertools
import numpy as np
import os
import pandas as pd
import random
import scipy.io
import time

from SpeechBCI import ElectrodeArray
from Sentence import Sentence

def block_zscore(flist, etl_dir):
    """
    Calculates global mean and std, then applies Z-scoring to the data.

    Args:
        flist (list): List of file identifiers.
        etl_dir (str): Directory containing the ETL files.

    Returns:
        tuple: Global mean and standard deviation.
    """
    global_data = np.array([])
    for idkey in flist:
        fname = os.path.join(etl_dir, idkey + '.csv')
        x = ElectrodeArray()
        x.load(fname)
        global_data = np.append(global_data, x.xt.flatten())
    
    global_mean = np.mean(global_data)
    global_std = np.std(global_data)
    
    for idkey in flist:
        fname = os.path.join(etl_dir, idkey + '.csv')
        x = ElectrodeArray()
        x.load(fname)
        x.xt = (x.xt - global_mean) / global_std
        x.save(etl_dir)
        
    return global_mean, global_std
    
def etl_blockZ(session_id, mat_data, etl_dir, stats_dir='', show_plots=False):
    """
    Performs ETL operations on a session.

    Args:
        session_id (str): Identifier for the session.
        mat_data (dict): Data loaded from a .mat file.
        etl_dir (str): Directory to save the ETL files.
        stats_dir (str, optional): Directory to save the statistics files. Defaults to ''.
        show_plots (bool, optional): Whether to generate and save plots. Defaults to False.

    Returns:
        tuple: Number of blocks, trials, and time samples.
    """
    num_trials = len(mat_data['sentenceText'])
    num_time_samples = sum([mat_data['spikePow'][0, icol].shape[0] for icol in range(num_trials)])
    max_block_id = max([int(mat_data['blockIdx'][i][0]) for i in range(num_trials)])
    num_blocks = len(set([int(mat_data['blockIdx'][i][0]) for i in range(num_trials)]))

    print(f"ETL on {session_id} with {num_trials} trials, {num_blocks} blocks, and {num_time_samples} time samples.")

    roster = []
    for trial_id in range(num_trials):
        block_id = int(mat_data['blockIdx'][trial_id][0])
        sentence_text = mat_data['sentenceText'][trial_id]
        num_samples_this_trial = mat_data['spikePow'][0, trial_id].shape[0]
        x = Sentence('sentenceText', session_id, max_block_id, block_id, trial_id, num_samples_this_trial, sentence_text)
        x.save(etl_dir)
        
        var_list_raw = ['spikePow', 'tx1']
        var_list_etl = []
        for var_name_raw in var_list_raw:
            var_name = "6v_Inf_" + var_name_raw
            var_list_etl.append(var_name)
            start_chan = 64
            end_chan = 128
            num_rows = 8
            num_cols = 8
            x = ElectrodeArray(var_name, session_id, max_block_id, block_id, trial_id,
                               mat_data[var_name_raw][0, trial_id], start_chan, end_chan, num_rows, num_cols)
            x.save(etl_dir)
            roster.append([x.idkey, block_id, var_name])
            
            if show_plots:
                fig = x.implot(0, x.num_samples, 1)
                fig.write_html(f'{etl_dir}/{x.idkey}_implot.html', auto_open=False)
                
                fig = x.tsplot(0, 8)
                fig.write_html(f'{etl_dir}/{x.idkey}_tsplot.html', auto_open=False)
                
    df = pd.DataFrame(roster, columns=['idkey', 'block_id', 'var_name'])
    zstats = []
    blks = df['block_id'].unique()
    for iblk in blks:
        for ivar in var_list_etl:
            df_block = df[(df['block_id'] == iblk) & (df['var_name'] == ivar)]
            global_mean, global_sd = block_zscore(list(df_block['idkey']), etl_dir)
            zstats.append([iblk, ivar, global_mean, global_sd])
    
    if stats_dir:
        zstats = pd.DataFrame(zstats, columns=['block_id', 'var_name', 'global_mean', 'global_sd'])
        pd.set_option('display.float_format', '{:.3f}'.format)
        zstats.to_csv(os.path.join(stats_dir, session_id + '_zstats.csv'),
                      index=False, float_format='%.4f', sep=',')

    return num_blocks, num_trials, num_time_samples

def main():
    """
    Main function to execute the ETL process on all identified .mat files.
    """
    script_name = 'etl'
    start_time = time.perf_counter()
    print('*** ' + script_name + ' - START ***')

    np.random.seed(42)
    random.seed(42)
    
    raw_data_dir = "/home/ubuntu/speechBCI/data/competitionData/train"
    etl_dir = "/home/ubuntu/speechBCI/data/competitionData/etl"
    stats_dir = "/home/ubuntu/speechBCI/data/competitionData/stats"
    
    mat_files = []
    for root, _, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith(".mat"):
                mat_files.append(os.path.join(root, file))
                
    tot_num_sessions = 0
    tot_num_blocks = 0
    tot_num_trials = 0
    tot_time_samples = 0
    
    for mat_file_path in mat_files:
        try:
            print(f"\nProcessing {mat_file_path}")
            mat_data = scipy.io.loadmat(mat_file_path)
            session_id = os.path.splitext(os.path.basename(mat_file_path))[0]
            num_blocks, num_trials, num_time_samples = etl_blockZ(session_id, mat_data, etl_dir, stats_dir, False)
            tot_num_sessions += 1
            tot_num_blocks += num_blocks
            tot_num_trials += num_trials
            tot_time_samples += num_time_samples
            print(f"Completed {mat_file_path}")

        except FileNotFoundError:
            print(f"Error: The file '{mat_file_path}' was not found.")
            exit(1)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            exit(1)

    print(f'Total number of sessions: {tot_num_sessions}')      
    print(f'Total number of blocks: {tot_num_blocks}')
    print(f'Total number of trials: {tot_num_trials}')
    print(f'Total number of time samples: {tot_time_samples}')
  
    print(f'\nTotal elapsed time:  %.4f seconds' % (time.perf_counter() - start_time))
    print('*** ' + script_name + ' - END ***')
            
if __name__ == '__main__':
    main()

