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
from torch.nn.modules.transformer import PositionalEncoding
from tqdm import tqdm
from etl.Sentence import Sentence
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
        max_seq_len,  # For embedding sequences
        max_label_seq_len,  # For label sequences
        padding_vector,
        transform=None,
        target_transform=None,
    ):
        # Create positional encoding once for the maximum sequence length
        self.pos_encoder = PositionalEncoding(padding_vector.shape[0], dropout=0.0)
        # Create a dummy input to get the positional encodings
        dummy_input = torch.zeros(max_seq_len, 1, padding_vector.shape[0])
        self.positional_encodings = self.pos_encoder(dummy_input).squeeze(1)  # Shape: [max_seq_len, embedding_dim]

        dataset = self.gen_dataset(
            embed_dir,
            etl_dir,
            max_seq_len,
            max_label_seq_len,
            padding_vector,
            model_type,
            tokenizer,
        )
        self.trial_ids = dataset["trial_ids"]
        self.samples = dataset["padded_samples"]
        self.padding_masks = dataset["sample_padding_masks"]
        self.labels = dataset["labels"]
        self.label_ids = dataset["padded_label_ids"]
        self.label_masks = dataset["label_padding_masks"]
        self.val_flag = dataset["val_flag"]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "trial_id": self.trial_ids[idx],
            "vqvae_embeddings": self.samples[idx].clone().detach(),
            "padding_masks": self.padding_masks[idx].clone().detach(),
            "positional_encodings": self.positional_encodings.clone().detach(),  # Use the pre-computed encodings
            "label_embeddings": self.label_ids[idx].clone().detach(),
            "label_padding_mask": self.label_masks[idx].clone().detach(),
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
        max_seq_len,  # For embedding sequences
        max_label_seq_len,  # For label sequences
        padding_vector,
        model_type,
        tokenizer,
    ):
        
        trial_ids = []
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
            tmp_ids, tmp_samples, tmp_labels, tmp_val_flag = self._get_trial_data(
                trial_dir, label_dir, trial_list, val_flag_value
            )
            trial_ids.extend(tmp_ids)
            samples.extend(tmp_samples)
            labels.extend(tmp_labels)
            val_flag.extend(tmp_val_flag)
            print(f"Found {len(tmp_samples)} samples for {sub_dir}.")

            #
            # Note: samples is a list of trial tensors, one T x E x H x W tensor per trial:
            #   T is input seqquence length
            #   E is the embedding dimension
            #   H x W is the number of embeddeing vectors
            # For each trial create a new sample tensor that is max_seq_len x (E*H*w).
            #       a.  Pad as necessary using padding_vector.
            #               We've made the choice here that the padding_vector is the average
            #               of the codebook vectors passed in as an argument. This keeps
            #               things very clean where we use ONLY training data - no peeking!
            #       b.  Generate a padding mask on the original T non-padded values.
            #       c.  Position encoding happens in the transformer model.

        padded_samples = []
        sample_padding_masks = []
        padded_label_ids = []
        label_padding_masks = []

        for sample, label in tqdm(zip(samples, labels), desc="Tokenizing Trial Data"):
            padded, mask = self._pad_sample(sample, max_seq_len, padding_vector)
            padded_samples.append(padded)
            sample_padding_masks.append(mask)

            # Pad label sequence
            encoded_labels = self._pad_label(
                label, max_label_seq_len, model_type, tokenizer
            )
            padded_label_ids.append(encoded_labels.input_ids.squeeze())
            label_padding_masks.append(encoded_labels.attention_mask.squeeze())

        #
        # Note that the padded samples and labels are now T x (E*H*W) tensors.
        # The padding masks are T x 1 tensors.
        # As a diagnostic, compute the min, max, average, and std of
        # the padding mask lengths, where padding mask means the
        # number of non-zero values in the mask.
        padding_mask_lengths = [mask.sum().item() for mask in sample_padding_masks]
        min_len = min(padding_mask_lengths)
        max_len = max(padding_mask_lengths)
        avg_len = np.mean(padding_mask_lengths)
        std_len = np.std(padding_mask_lengths)
        print(
            f"Embedding Mask Stats: min: {min_len}, max: {max_len}, avg: {avg_len}, std: {std_len}"
        )

        label_mask_lengths = [mask.sum().item() for mask in label_padding_masks]
        min_len = min(label_mask_lengths)
        max_len = max(label_mask_lengths)
        avg_len = np.mean(label_mask_lengths)
        std_len = np.std(label_mask_lengths)
        print(
            f"Label Mask Stats: min: {min_len}, max: {max_len}, avg: {avg_len}, std: {std_len}"
        )

        return {
            "trial_ids": trial_ids,
            "padded_samples": padded_samples,
            "sample_padding_masks": sample_padding_masks,
            "labels": labels,
            "padded_label_ids": padded_label_ids,
            "label_padding_masks": label_padding_masks,
            "val_flag": val_flag,
        }

    @staticmethod
    def _get_trial_data(trial_dir, label_dir, trial_list, val_flag_value):

        trial_ids = []
        samples = []
        labels = []
        val_flag = []
        max_input_seq_len = 0

        for trial in trial_list:
            trial_data = torch.load(
                os.path.join(trial_dir, f"{trial}.pt"), weights_only=True
            )
            text_data = Sentence()
            text_data.load(os.path.join(label_dir, f"{trial}_sentenceText.csv"))
            trial_ids.append(trial)
            samples.append(trial_data)
            labels.append(text_data.sentence_txt)
            val_flag.append(val_flag_value)
            if trial_data.shape[0] > max_input_seq_len:
                max_input_seq_len = trial_data.shape[0]
        print(
            f"Generated {len(samples)} samples with max input length {max_input_seq_len}."
        )
        return trial_ids, samples, labels, val_flag

    @staticmethod
    def _pad_sample(sample, max_seq_len, padding_vector):
        seq_len = sample.shape[0]
        h = sample.shape[2]
        w = sample.shape[3]
        if seq_len > max_seq_len:
            print(
                f"\nWarning _pad_sample: sample len {seq_len} > max_seq_len {max_seq_len}."
            )
            sample = sample[:max_seq_len]
            seq_len = max_seq_len

        # Sample data is coming in as T x E x H x W
        # We want to flatten to T x (E*H*W)
        sample = sample.view(seq_len, -1)

        # The padding vector is coming in as [E]
        # 1. First, repeat it H*W times to get a single flattened row
        flattened_padding = padding_vector.repeat(h * w)  # Shape: [E*H*W]

        # 2. Create a new tensor filled with the padding value
        # Use repeat() instead of expand() to actually allocate new memory
        padded_sample = flattened_padding.unsqueeze(0).repeat(
            max_seq_len, 1
        )  # Shape: [max_seq_len, E*H*W]

        # Copy original data (or truncate)
        copy_len = min(seq_len, max_seq_len)
        padded_sample[:copy_len] = sample[:copy_len]

        # Create padding mask (1 for real data, 0 for padding)
        mask = torch.zeros(max_seq_len, dtype=torch.bool)
        mask[:copy_len] = 1

        return padded_sample, mask

    @staticmethod
    def _pad_label(label, max_length, model_type, tokenizer):
        """Pad label to max_length using tokenizer."""
        encoded_labels = tokenizer(
            label,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded_labels  # Return the tokenizer output directly

    def preprocess_target_text(self, text, model_type):
        """Preprocess target text with appropriate sentence markers based on model type."""
        if model_type == "bart":
            # For BART: we'll wrap the target text in sentence tags
            # BART already adds <s> and </s> tokens, but we want sentence-level markers
            return f"<sentence> {text} </sentence>"
        else:
            # For T5 or other models, return unchanged
            return text
