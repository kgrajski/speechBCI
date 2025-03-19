#
# 19March2025 - BART implementation
# Sequence: etl.py -> main_vqvae3D.py (training) -> main_vqvae3D.py (encoding) -> main_mmllm_bart.py
#

import collections
import math
import numpy as np
import os
import string
from tabulate import tabulate
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from Sentence import Sentence
from transformers import BartTokenizer

class SpeechBCIDataSet_Embedded_BART(Dataset):
    
    # Class-level tokenizer variable - initialized only once
    _tokenizer = None
    
    def __init__(self, label_dir, embeddings_dir, max_seq_len, padding_vector,
                 transform=None, target_transform=None):
        self.samples, self.attention_masks, \
        self.labels, self.label_ids, self.label_masks, \
        self.val_flag = self.gen_dataset(label_dir, embeddings_dir, max_seq_len, padding_vector)
        
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return {
            "vqvae_embeddings": self.samples[idx],
            "attention_mask": self.attention_masks[idx],
            "label_embeddings": self.label_ids[idx],
            "label_attention_mask": self.label_masks[idx]
        }
    
    def gen_dataset(self, label_dir, embeddings_dir, max_seq_len, padding_vector):
        samples = []
        labels = []
        val_flag = []
        
        # Get embedded trial data for training and testing sets
        for sub_dir in ['train', 'test']:
            # Locate the trial embedded data and labels
            trial_dir = os.path.join(embeddings_dir, sub_dir)
            trial_list = [f.split('.')[0] for f in os.listdir(trial_dir) if f.endswith('.pt')]
            val_flag_value = sub_dir == 'test'
            
            # Collect all of the trial embedded data and labels
            tmp_samples, tmp_labels, tmp_val_flag = self._get_trial_data(trial_dir, label_dir, trial_list, val_flag_value)
            samples.extend(tmp_samples)
            labels.extend(tmp_labels)
            val_flag.extend(tmp_val_flag)
            print(f"Generated {len(tmp_samples)} samples for {sub_dir}.")

        padded_samples = []
        sample_attention_masks = []
        padded_label_ids = []
        label_attention_masks = []
        
        for sample, label in tqdm(zip(samples, labels), desc="Tokenizing Trial Data"):
            # Pad samples
            padded, mask = self._pad_sample(sample, max_seq_len, padding_vector)
            padded_samples.append(padded)
            sample_attention_masks.append(mask)
            
            # Process labels with BART tokenizer
            padded, mask = self._pad_label_bart(label, max_seq_len)
            padded_label_ids.append(padded.squeeze())
            label_attention_masks.append(mask.squeeze())
            
        return padded_samples, sample_attention_masks, labels, padded_label_ids, label_attention_masks, val_flag
    
    @staticmethod
    def _get_trial_data(trial_dir, label_dir, trial_list, val_flag_value):
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
    
    @staticmethod
    def _pad_sample(sample, max_seq_len, padding_vector):
        seq_len = sample.shape[0]
        if seq_len > max_seq_len:
            print(f"Warning _pad_sample: sample len {seq_len} > max_seq_len {max_seq_len}.")
            seq_len = max_seq_len
            
        # Create new padded tensor
        padded_sample = padding_vector.repeat(max_seq_len, 1)
        
        # Copy original data (or truncate)
        copy_len = min(seq_len, max_seq_len)
        padded_sample[:copy_len] = sample[:copy_len]
        
        # Create attention mask (1 for real data, 0 for padding)
        mask = torch.zeros(max_seq_len, dtype=torch.bool)
        mask[:copy_len] = 1
        
        return padded_sample, mask
    
    @classmethod
    def _pad_label_bart(cls, labels, max_seq_len):
        # Initialize tokenizer
        if cls._tokenizer is None:
            cls._tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            
        # BART uses different padding and eos tokens than T5
        encoded_labels = cls._tokenizer(
            labels,
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        return encoded_labels.input_ids, encoded_labels.attention_mask
   
    def _train_test_label_compare(self):
        """Generates and prints statistics about training and test datasets"""
        # Split labels into training and testing sets
        train_labels = [label for label, flag in zip(self.labels, self.val_flag) if not flag]
        test_labels = [label for label, flag in zip(self.labels, self.val_flag) if flag]

        # Get statistics
        train_stats = self._get_label_statistics(train_labels)
        test_stats = self._get_label_statistics(test_labels)

        unique_train_words = train_stats['unique_words']
        unique_test_words = test_stats['unique_words']
        
        # Calculate common, unique to training, and unique to testing words
        common_words = unique_train_words & unique_test_words
        words_unique_to_train = unique_train_words - unique_test_words
        words_unique_to_test = unique_test_words - unique_train_words
        
        # Calculate word frequencies
        common_word_counts_train = {word: train_stats['words'].count(word) for word in common_words}
        common_word_counts_test = {word: test_stats['words'].count(word) for word in common_words}
        
        # Get top 20 words for different categories
        top_20_common_words_train = collections.Counter(common_word_counts_train).most_common(20)
        top_20_common_words_test = collections.Counter(common_word_counts_test).most_common(20)
        top_20_unique_train_words = collections.Counter(
            {word: train_stats['words'].count(word) for word in words_unique_to_train}
        ).most_common(20)
        top_20_unique_test_words = collections.Counter(
            {word: test_stats['words'].count(word) for word in words_unique_to_test}
        ).most_common(20)
        
        # Find maximum lengths
        max_embedded_label_len = max([mask.sum() for mask in self.label_masks])
        max_attention_mask_len = max([mask.sum() for mask in self.attention_masks])

        # Compile statistics
        label_stats = {
            'train_stats': train_stats,
            'test_stats': test_stats,
            'common_words_count': len(common_words),
            'unique_train_words_count': len(words_unique_to_train),
            'unique_test_words_count': len(words_unique_to_test),
            'top_20_common_words_train': top_20_common_words_train,
            'top_20_common_words_test': top_20_common_words_test,
            'top_20_unique_train_words': top_20_unique_train_words,
            'top_20_unique_test_words': top_20_unique_test_words,
            'max_embedded_label_len': max_embedded_label_len,
            'max_attention_mask_len': max_attention_mask_len
        }
        self._pretty_print_label_stats(label_stats)

    @staticmethod
    def _get_label_statistics(labels):
        # Standard statistics generation (unchanged from original)
        translator = str.maketrans('', '', string.punctuation)
        cleaned_labels = [label.translate(translator).lower() for label in labels]
        
        longest_label_length = len(max(cleaned_labels, key=len))
        shortest_label_length = len(min(cleaned_labels, key=len))

        words = [word for label in cleaned_labels for word in label.split()]
        word_counts = collections.Counter(words)
        unique_words = set(words)
        avg_sentence_length = sum(len(label.split()) for label in cleaned_labels) / len(cleaned_labels)
        top_20_words = word_counts.most_common(20)
        bottom_20_words = word_counts.most_common()[:-21:-1]
        total_words = len(words)

        return {
            'longest_label_length': longest_label_length,
            'shortest_label_length': shortest_label_length,
            'words': words,
            'unique_words': unique_words,
            'unique_words_count': len(unique_words),
            'top_20_words': top_20_words,
            'bottom_20_words': bottom_20_words,
            'avg_sentence_length': avg_sentence_length,
            'total_words': total_words
        }
    
    @staticmethod
    def _pretty_print_label_stats(label_stats):
        # Pretty printing of statistics (unchanged from original)
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
        
        print("\nMax Embedded Label Length:", label_stats['max_embedded_label_len'])
        print("Max Attention Mask Length:", label_stats['max_attention_mask_len'])