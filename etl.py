
import itertools
import numpy as np
import os
import pandas as pd
import random
import scipy.io
import time

#
# Get the SpeechBCI class defs.
#
from SpeechBCI import ElectrodeArray
from Sentence import Sentence

#
# Local Code
#
def block_zscore(flist, etl_dir):

    # First pass: Calculate global mean and std
    global_data = np.array([])
    for idkey in flist:
        fname = os.path.join(etl_dir + os.sep + idkey + '.csv')
        x = ElectrodeArray()
        x.load(fname)
        global_data = np.append(global_data, x.xt.flatten())
    
    global_mean = np.mean(global_data)
    global_std = np.std(global_data)
    
    # Second pass: Z-score the data using global mean and std
    for idkey in flist:
        fname = os.path.join(etl_dir + os.sep + idkey + '.csv')
        x = ElectrodeArray()
        x.load(fname)
        x.xt = (x.xt - global_mean) / global_std
        x.save(etl_dir)
        
    return global_mean, global_std
    
def etl_blockZ(session_id, mat_data, etl_dir, stats_dir = '', show_plots=False):
        #
        # Detailed structure of mat_data is desribed elsewhere:
        # See: https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq
        # Consequently, we can hard-wire a couple of things.
        # 
        # The raw data combines measurement data from electrods placed at four disctinct
        # locations.  We will create labelled subsets accoringly.  Such subsets will
        # be defined by the start_chan and end_chan and a label.
        #
        
        # The number of sentences tells the number of trials.
    num_trials = len(mat_data['sentenceText'])
    num_time_samples = sum([mat_data['spikePow'][0,icol].shape[0] for icol in range(num_trials)])
    
        # We'd like to know the highest block index in the session, because later
        # those will be used as the source for test and/or validation data.
        # Further note that in this dataset, the block_id is 1-based.
    max_block_id = max([int(mat_data['blockIdx'][i][0]) for i in range(num_trials)])
    num_blocks = len(set([int(mat_data['blockIdx'][i][0]) for i in range(num_trials)]))

    print(f"ETL on {session_id} with {num_trials} trials, {num_blocks} blocks, and {num_time_samples} time samples.")

        # Iterate over the trials.
        # Every trial will have same session_id.
        # Every trial will generate two objects: Sentence and ElectrodeArray.
        # The ElectrodeArray objects will be created for spikePow and tx1.
        # Further, we will create a "superior" and "inferior" 6v subset for each ElectrodeArray.
        
        # To do the block-based Z-scoring, we need to know the number of samples in each trial.
    roster = []
    for trial_id in range(num_trials):
        block_id = int(mat_data['blockIdx'][trial_id][0])

            # Process the sentence text.
        sentence_text = mat_data['sentenceText'][trial_id]
        num_samples_this_trial = mat_data['spikePow'][0,trial_id].shape[0]
        x = Sentence('sentenceText', session_id, max_block_id, block_id, trial_id, num_samples_this_trial, sentence_text)
        x.save(etl_dir)
        
            # Process the electrode arrays.
            # Hard-wired knowledge is indicated by the default values.
        var_list_raw = ['spikePow', 'tx1']
        var_list_etl = []
        for var_name_raw in var_list_raw:
                # Generate the data for area 6v Inferior.
                # Per Willett, et al. (2023). 6v Inf had better speech decoding performance.
                # We can revisit and reconfirm in a follow-up study.
            var_name = "6v_Inf_" + var_name_raw
            var_list_etl.append(var_name)
            start_chan = 64
            end_chan = 128
            num_rows = 8
            num_cols = 8
            x = ElectrodeArray(var_name, session_id, max_block_id, block_id, trial_id,
                               mat_data[var_name_raw][0,trial_id], start_chan, end_chan, num_rows, num_cols)
            x.save(etl_dir)
            roster.append([x.idkey, block_id, var_name])
            
            if show_plots:
                fig = x.implot(0, x.num_samples, 1)
                fig.write_html(f'{etl_dir}/{x.idkey}_implot.html', auto_open=False)
                
                fig = x.tsplot(0, 8)
                fig.write_html(f'{etl_dir}/{x.idkey}_tsplot.html', auto_open=False)
                
    #
    # Do the block-based Z-scoring.
    #
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

#
# MAIN
#
def main():

        # Set script name for console log
    script_name = 'etl'
    
        # Start timer
    start_time = time.perf_counter()
    print('*** ' + script_name + ' - START ***')

        # For reproducibility
    np.random.seed(42)  # Set seed for reproducibility.
    random.seed(42)  # Set seed for reproducibility.
    
        #
        # Directory info.
        #
    raw_data_dir = "/home/ubuntu/speechBCI/data/competitionData/train"
    etl_dir = "/home/ubuntu/speechBCI/data/competitionData/etl"
    stats_dir = "/home/ubuntu/speechBCI/data/competitionData/stats"
    
        #
        # Get the list of .mat file names.
        # This is the Competition Data from Willett, et al. (2023).
        #
    mat_files = []
    for root, _, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith(".mat"):
                mat_files.append(os.path.join(root, file))
                
        #
        # Perform ETL on each of the identified .mat files.
        #
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
  
        # Wrap-up
    print(f'\nTotal elapsed time:  %.4f seconds' % (time.perf_counter() - start_time))
    print('*** ' + script_name + ' - END ***')
            
#
# EXECUTE
#
if __name__ == '__main__':
    main()

