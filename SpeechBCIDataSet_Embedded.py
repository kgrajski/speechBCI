"""
This module defines the SpeechBCIDataSet_Embedded class, a custom PyTorch Dataset
designed for handling Speech BCI Array Recordings that have gone through ETL and
VQ-VAE embedding.

Main differences from SpeechBCIDataSet_3D and SpeechBCIDataSet_2D is that the data
is now embedded in a lower-dimensional space and stored as one file per trial.

Classes:
    SpeechBCIDataSet_Embedded: A custom PyTorch Dataset for Speech BCI Array Recordings.

Usage example:
    dataset = SpeechBCIDataSet_Embedded(label_dir="/path/to/labels", embeddings_dir="/path/to/embeddings")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in dataloader:
        # process data
"""

#
# 14March2025 - actively working.
# Sequence: etl.py -> main_vqvae3D.py (training) -> main_vqvae3D.py (encoding) -> main_mmllm.py
#

import collections
import numpy as np
import os
import string
import plotly.express as px
import pandas as pd
from tabulate import tabulate
import torch
from torch.utils.data import Dataset
from Sentence import Sentence

class SpeechBCIDataSet_Embedded(Dataset):
    """
    PyTorch custom Dataset tuned for Speech BCI Array Recordings.
    
    We treat each trial as a 3D array: time series of 2D images.
    
    Note:
        Adhere to the convention of NTCHW.

    Args:
        label_dir (str): Directory containing the label files.
        embeddings_dir (str): Directory containing the embedded trial data files.
        transform (callable, optional): Optional transform to be applied on a sample.
        target_transform (callable, optional): Optional transform to be applied on the target.
    """
    
    def __init__(self, label_dir, embeddings_dir, transform=None, target_transform=None):
        self.samples, self.labels, self.val_flag = self.gen_dataset(label_dir, embeddings_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def _get_trial_data(trial_dir, label_dir, trial_list, val_flag_value):
        """
        Helper function to get trial data and labels.

        Args:
            trial_dir (str): Directory containing the trial data files.
            label_dir (str): Directory containing the label (sentence text) files.
            trial_list (list): List of trial identifiers.
            val_flag_value (bool): Flag indicating if the data is from the validation set.

        Returns:
            tuple: A tuple containing:
                - list: List of trial data samples.
                - list: List of trial labels.
                - list: List of validation set flags.
        """
        samples = []
        labels = []
        val_flag = []
        for trial in trial_list:
            trial_data = torch.load(os.path.join(trial_dir, f"{trial}.pt"), weights_only=True)
            text_data = Sentence()
            text_data.load(os.path.join(label_dir, f"{trial}_sentenceText.csv"))
            samples.append(trial_data)
            labels.append(text_data.sentence_txt)
            val_flag.append(val_flag_value)
        return samples, labels, val_flag

    def gen_dataset(self, label_dir, embeddings_dir):
        """
        Generate the dataset by reading label and embedding files from the specified directories.

        Args:
            label_dir (str): Directory containing the label files.
            embeddings_dir (str): Directory containing the embedded trial data files.

        Returns:
            tuple: A tuple containing:
                - list: List of samples.
                - list: List of labels.
                - list: List of validation set flags.
        """
        samples = []
        labels = []
        val_flag = []
        
        for sub_dir in ['train', 'test']:
            trial_dir = os.path.join(embeddings_dir, sub_dir)
            trial_list = [f.split('.')[0] for f in os.listdir(trial_dir) if f.endswith('.pt')]
            val_flag_value = sub_dir == 'test'
            tmp_samples, tmp_labels, tmp_val_flag = self._get_trial_data(trial_dir, label_dir, trial_list, val_flag_value)
            samples.extend(tmp_samples)
            labels.extend(tmp_labels)
            val_flag.extend(tmp_val_flag)
            print(f"Generated {len(tmp_samples)} samples for {sub_dir}.")
            
        return samples, labels, val_flag
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The sample at the specified index.
                - str: The label for the sample (sentence text).
        """
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = self.labels[idx]
        return x, y

    @staticmethod
    def _get_label_statistics(labels):
        """
        Provides basic descriptive statistics for the labels.

        Args:
            labels (list of str): List of label strings.

        Returns:
            dict: Dictionary containing various statistics.
        """
        # Remove punctuation and convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        cleaned_labels = [label.translate(translator).lower() for label in labels]

        # Split labels into words
        words = [word for label in cleaned_labels for word in label.split()]

        # Count word frequencies
        word_counts = collections.Counter(words)

        # Get unique words
        unique_words = set(words)

        # Calculate average sentence length
        avg_sentence_length = sum(len(label.split()) for label in cleaned_labels) / len(cleaned_labels)

        # Get top 20 most frequent words
        top_20_words = word_counts.most_common(20)

        # Get bottom 20 least frequent words
        bottom_20_words = word_counts.most_common()[:-21:-1]

        # Calculate total number of words
        total_words = len(words)

        return {
            'words': words,
            'unique_words': unique_words,
            'unique_words_count': len(unique_words),
            'top_20_words': top_20_words,
            'bottom_20_words': bottom_20_words,
            'avg_sentence_length': avg_sentence_length,
            'total_words': total_words
        }
    
    def _train_test_label_compare(self):
            """
            Compares label statistics between training and testing sets.

            Returns:
                dict: Dictionary containing various statistics and comparisons.
            """
            # Split labels into training and testing sets
            train_labels = [label for label, flag in zip(self.labels, self.val_flag) if not flag]
            test_labels = [label for label, flag in zip(self.labels, self.val_flag) if flag]

            # Get statistics for training and testing sets
            # The statistics include the number of words, unique words, and the words themselves
            # And the labels are "cleaned" so use these for further stats.
            train_stats = self._get_label_statistics(train_labels)
            test_stats = self._get_label_statistics(test_labels)

            unique_train_words = train_stats['unique_words']
            unique_test_words = test_stats['unique_words']
            
            # Calculate common, unique to training, and unique to testing words
            common_words = unique_train_words & unique_test_words
            words_unique_to_train = unique_train_words - unique_test_words
            words_unique_to_test = unique_test_words - unique_train_words
            
            # For the common_words, what is their frequency in the training and testing sets?
            common_word_counts_train = {word: train_stats['words'].count(word) for word in common_words}
            common_word_counts_test = {word: test_stats['words'].count(word) for word in common_words}
            
            # Get top 20 common words in training and testing sets
            top_20_common_words_train = collections.Counter(common_word_counts_train).most_common(20)
            top_20_common_words_test = collections.Counter(common_word_counts_test).most_common(20)
            
            # Get top 20 unique words in training and testing sets
            top_20_unique_train_words = collections.Counter(
                {word: train_stats['words'].count(word) for word in words_unique_to_train}
            ).most_common(20)
            top_20_unique_test_words = collections.Counter(
                {word: test_stats['words'].count(word) for word in words_unique_to_test}
            ).most_common(20)

            label_stats = {
                'train_stats': train_stats,
                'test_stats': test_stats,
                'common_words_count': len(common_words),
                'unique_train_words_count': len(unique_train_words),
                'unique_test_words_count': len(unique_test_words),
                'top_20_common_words_train': top_20_common_words_train,
                'top_20_common_words_test': top_20_common_words_test,
                'top_20_unique_train_words': top_20_unique_train_words,
                'top_20_unique_test_words': top_20_unique_test_words
            }
            self._pretty_print_label_stats(label_stats)

    @staticmethod
    def _pretty_print_label_stats(label_stats):
        """
        Pretty prints the contents of the label_stats dictionary.

        Args:
            label_stats (dict): Dictionary containing various statistics and comparisons.
        """
        # Exclude 'unique_words' and 'words' keys from train_stats and test_stats
        train_stats_filtered = {k: v for k, v in label_stats['train_stats'].items() if k not in ['unique_words', 'words']}
        test_stats_filtered = {k: v for k, v in label_stats['test_stats'].items() if k not in ['unique_words', 'words']}

        print("Training Set Statistics:")
        print(tabulate(train_stats_filtered.items(), headers=['Statistic', 'Value'], tablefmt='grid'))

        print("\nTesting Set Statistics:")
        print(tabulate(test_stats_filtered.items(), headers=['Statistic', 'Value'], tablefmt='grid'))

        print("\nCommon Words Count:", label_stats['common_words_count'])
        print("Unique Train Words Count:", label_stats['unique_train_words_count'])
        print("Unique Test Words Count:", label_stats['unique_test_words_count'])

        print("\nTop 20 Common Words in Training Set:")
        print(tabulate(label_stats['top_20_common_words_train'], headers=['Word', 'Count'], tablefmt='grid'))

        print("\nTop 20 Common Words in Testing Set:")
        print(tabulate(label_stats['top_20_common_words_test'], headers=['Word', 'Count'], tablefmt='grid'))

        print("\nTop 20 Unique Train Words:")
        print(tabulate(label_stats['top_20_unique_train_words'], headers=['Word', 'Count'], tablefmt='grid'))

        print("\nTop 20 Unique Test Words:")
        print(tabulate(label_stats['top_20_unique_test_words'], headers=['Word', 'Count'], tablefmt='grid'))

    def visualize_word_frequencies(self, n_words=50, split='all'):
        """
        Visualize word frequencies using plotly express.
        
        Args:
            n_words (int): Number of top words to visualize
            split (str): Which data split to visualize ('train', 'test', or 'all')
        
        Returns:
            plotly.graph_objects.Figure: Histogram figure of word frequencies
        """
        if split == 'train':
            labels = [label for label, flag in zip(self.labels, self.val_flag) if not flag]
        elif split == 'test':
            labels = [label for label, flag in zip(self.labels, self.val_flag) if flag]
        else:
            labels = self.labels
        
        # Clean and tokenize
        translator = str.maketrans('', '', string.punctuation)
        cleaned_labels = [label.translate(translator).lower() for label in labels]
        words = [word for label in cleaned_labels for word in label.split()]
        
        # Count word frequencies
        word_counts = collections.Counter(words)
        top_words = word_counts.most_common(n_words)
        
        # Create a DataFrame for plotting
        df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
        
        # Create the histogram
        fig = px.histogram(df, x='Word', y='Frequency', title=f'Top {n_words} Word Frequencies ({split})')
        fig.update_layout(xaxis_title='Word', yaxis_title='Frequency', xaxis={'categoryorder': 'total descending'})
        
        return fig
    
    def visualize_sentence_lengths(self):
        """
        Visualize distribution of sentence lengths using plotly express.
        
        Returns:
            plotly.graph_objects.Figure: Histogram figure of sentence lengths
        """
        # Calculate sentence lengths
        train_lengths = [len(label.split()) for label, flag in zip(self.labels, self.val_flag) if not flag]
        test_lengths = [len(label.split()) for label, flag in zip(self.labels, self.val_flag) if flag]
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'Length': train_lengths + test_lengths,
            'Split': ['Train'] * len(train_lengths) + ['Test'] * len(test_lengths)
        })
        
        # Create the histogram
        fig = px.histogram(
            df, x='Length', color='Split', 
            marginal='box',
            title='Distribution of Sentence Lengths',
            nbins=30
        )
        fig.update_layout(xaxis_title='Sentence Length (words)', yaxis_title='Count')
        
        return fig
    
    def visualize_embedding_stats(self):
        """
        Visualize statistics of the embeddings using plotly express.
        
        Returns:
            dict: Dictionary containing various plotly figures
        """
        figures = {}
        
        # Get embedding dimensions for a few samples
        sample_embeddings = [self.samples[i] for i in range(min(10, len(self.samples)))]
        
        # Plot embedding lengths
        lengths = [len(emb) for emb in self.samples]
        df_lengths = pd.DataFrame({'Sequence Length': lengths})
        
        figures['length_dist'] = px.histogram(
            df_lengths, x='Sequence Length',
            title='Distribution of Embedding Sequence Lengths',
            nbins=30
        )
        
        # Plot embedding means and std devs
        means = [torch.mean(torch.tensor(emb)).item() for emb in self.samples]
        stds = [torch.std(torch.tensor(emb)).item() for emb in self.samples]
        
        df_stats = pd.DataFrame({
            'Mean': means,
            'StdDev': stds
        })
        
        figures['means'] = px.histogram(df_stats, x='Mean', title='Distribution of Embedding Means')
        figures['stds'] = px.histogram(df_stats, x='StdDev', title='Distribution of Embedding Standard Deviations')
        
        # Scatter plot of means vs stds
        figures['means_vs_stds'] = px.scatter(
            df_stats, x='Mean', y='StdDev',
            title='Mean vs Standard Deviation of Embeddings'
        )
        
        return figures
