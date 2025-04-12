#
# 18March2025 - actively working.
# Sequence: etl.py -> main_vqvae3D.py (training) -> main_vqvae3D.py (encoding) -> main_mmllm.py
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
from transformers import T5Tokenizer


class SpeechBCIDataSet_Embedded(Dataset):

    #
    # The required basic methods: __init__, __len__, and __getitem__.
    #
    def __init__(
        self,
        embed_dir,
        etl_dir,
        tokenizer,
        model_type,
        max_seq_len,
        padding_vector,
        transform=None,
        target_transform=None,
    ):
        (
            self.samples,
            self.attention_masks,
            self.labels,
            self.label_ids,
            self.label_masks,
            self.val_flag,
        ) = self.gen_dataset(
            embed_dir, etl_dir, max_seq_len, padding_vector, model_type, tokenizer
        )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "vqvae_embeddings": self.samples[idx].clone().detach(),
            "attention_mask": self.attention_masks[idx].clone().detach(),
            "label_embeddings": self.label_ids[idx].clone().detach(),
            "label_attention_mask": self.label_masks[idx].clone().detach(),
            "original_text": self.labels[idx],  # Include original text
        }

    #
    # The following methods are used to generate the dataset.
    # Preference is to load all of the data into memory at once.
    #

    def gen_dataset(
        self,
        embeddings_dir,
        label_dir,
        max_seq_len,
        padding_vector,
        model_type,
        tokenizer,
    ):

        samples = []
        labels = []
        val_flag = []

        #
        # Get embedded trial data for training and testing sets.
        # Recall that for this dataset, they are prescribed. So we loop over both.
        #
        for sub_dir in ["train", "test"]:

            # Locate the trial embedded data and labels
            trial_dir = os.path.join(embeddings_dir, sub_dir)
            trial_list = [
                f.split(".")[0] for f in os.listdir(trial_dir) if f.endswith(".pt")
            ]
            val_flag_value = sub_dir == "test"

            # Collect all of the trial embedded data and labels
            tmp_samples, tmp_labels, tmp_val_flag = self._get_trial_data(
                trial_dir, label_dir, trial_list, val_flag_value
            )
            samples.extend(tmp_samples)
            labels.extend(tmp_labels)
            val_flag.extend(tmp_val_flag)
            print(f"Generated {len(tmp_samples)} samples for {sub_dir}.")

            #
            # Note: samples is a list of trial tensors, one T x E tensor per trial:
            #   T is input seqquence length
            #   E is the embedding dimension
            # For each trial create a new sample tensor that is max_seq_len x E.
            #       a.  Pad as necessary using padding_vector.
            #               We've made the choice here that the padding_vector is the average
            #               of the codebook vectors passed in as an argument. This keeps
            #               things very clean where we use ONLY training data - no peeking!
            #       b.  Generate an attention mask on the original T non-padded values.
            #       c.  Position encoding happens in the transformer model.

        padded_samples = []
        sample_attention_masks = []
        padded_label_ids = []
        label_attention_masks = []

        for sample, label in tqdm(zip(samples, labels), desc="Tokenizing Trial Data"):

            # If introducing new model types beyond T5 and BART, may need to
            # adjust the padding and attention mask handling in light of any
            # possible input length requirements.
            padded, mask = self._pad_sample(sample, max_seq_len, padding_vector)
            padded_samples.append(padded)
            sample_attention_masks.append(mask)

            # Label padding is model-specific, but details are in the method.
            padded, mask = self._pad_label(label, max_seq_len, model_type, tokenizer)
            padded_label_ids.append(padded.squeeze())
            label_attention_masks.append(mask.squeeze())

        return (
            padded_samples,
            sample_attention_masks,
            labels,
            padded_label_ids,
            label_attention_masks,
            val_flag,
        )

    @staticmethod
    def _get_trial_data(trial_dir, label_dir, trial_list, val_flag_value):

        samples = []
        labels = []
        val_flag = []
        # Load each trial's data and labels
        max_input_seq_len = 0
        for trial in trial_list:
            trial_data = torch.load(
                os.path.join(trial_dir, f"{trial}.pt"), weights_only=True
            )
            text_data = Sentence()
            text_data.load(os.path.join(label_dir, f"{trial}_sentenceText.csv"))
            samples.append(trial_data)
            labels.append(text_data.sentence_txt)
            val_flag.append(val_flag_value)
            if trial_data.shape[0] > max_input_seq_len:
                max_input_seq_len = trial_data.shape[0]
        print(
            f"Max input sequence length for trial data: {max_input_seq_len}."
        )
        return samples, labels, val_flag

    @staticmethod
    def _pad_sample(sample, max_seq_len, padding_vector):
        seq_len = sample.shape[0]
        if seq_len > max_seq_len:
            print(
                f"Warning _pad_sample: sample len {seq_len} > max_seq_len {max_seq_len}."
            )
            seq_len = max_seq_len

            # Create new padded tensor
        padded_sample = padding_vector.repeat(max_seq_len, 1)

        # copy original data (or truncate)
        copy_len = min(seq_len, max_seq_len)
        padded_sample[:copy_len] = sample[:copy_len]

        # Create attention mask (1 for real data, 0 for padding)
        mask = torch.zeros(max_seq_len, dtype=torch.bool)
        mask[:copy_len] = 1

        return padded_sample, mask

    @staticmethod
    def _pad_label(label, max_seq_len, model_type, tokenizer):
        """Tokenize and pad label text with model-specific handling."""
        # Apply sentence boundary tokens for BART
        if model_type.lower() == "bart":
            processed_label = f"<sentence> {label} </sentence>"
        else:
            processed_label = label
        
        # Tokenize with appropriate parameters
        encoded_labels = tokenizer(
            processed_label,
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        
        # Handle padding token IDs based on model type
        if model_type.lower() == "t5":
            input_ids = encoded_labels.input_ids
            input_ids[input_ids == tokenizer.pad_token_id] = -100
        else:
            input_ids = encoded_labels.input_ids
            
        return input_ids, encoded_labels.attention_mask

    def preprocess_target_text(self, text, model_type):
        """Preprocess target text with appropriate sentence markers based on model type."""
        if model_type == "bart":
            # For BART: we'll wrap the target text in sentence tags
            # BART already adds <s> and </s> tokens, but we want sentence-level markers
            return f"<sentence> {text} </sentence>"
        else:
            # For T5 or other models, return unchanged
            return text
