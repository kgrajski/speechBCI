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

    # Class-level tokenizer variable - initialized only once
    _tokenizer = None

    #
    # The required basic methods: __init__, __len__, and __getitem__.
    #
    def __init__(
        self,
        label_dir,
        embeddings_dir,
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
        ) = self.gen_dataset(label_dir, embeddings_dir, max_seq_len, padding_vector)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "vqvae_embeddings": self.samples[idx].clone().detach(),
            "attention_mask": self.attention_masks[idx].clone().detach(),
            "label_embeddings": self.label_ids[idx]
            .clone()
            .detach(),  # Changed key name to match T5 expectations
            "label_attention_mask": self.label_masks[idx].clone().detach(),
        }

        #
        # Additional methods to transform embedded data for use in fine-tuning an LLM.
        # Trying to be LLM-agnostic, but at a certain level that files, such as with
        # respect to tokenizers and other requirements.
        #

    def gen_dataset(self, label_dir, embeddings_dir, max_seq_len, padding_vector):

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

            padded, mask = self._pad_sample(sample, max_seq_len, padding_vector)
            padded_samples.append(padded)
            sample_attention_masks.append(mask)

            # MVP approach says let this model-specific call sit here for now.
            padded, mask = self._pad_label_t5(label, max_seq_len)
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
        for trial in trial_list:
            trial_data = torch.load(
                os.path.join(trial_dir, f"{trial}.pt"), weights_only=True
            )
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

    @classmethod
    def _pad_label_t5(cls, labels, max_seq_len):

        # Initialize tokenizer
        if cls._tokenizer is None:
            cls._tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=True)

        encoded_labels = cls._tokenizer(
            labels,
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        return encoded_labels.input_ids, encoded_labels.attention_mask

        #
        # The following methods process the raw label data.  Mainly
        # used during data exploration phase.
        #

    @staticmethod
    def _get_label_statistics(labels):

        # Remove punctuation and convert to lowercase
        translator = str.maketrans("", "", string.punctuation)
        cleaned_labels = [label.translate(translator).lower() for label in labels]

        # Find the length of the longest and shortest labels
        longest_label_length = len(max(cleaned_labels, key=len))
        shortest_label_length = len(min(cleaned_labels, key=len))

        # Split labels into words
        words = [word for label in cleaned_labels for word in label.split()]

        # Count word frequencies
        word_counts = collections.Counter(words)

        # Get unique words
        unique_words = set(words)

        # Calculate average sentence length
        avg_sentence_length = sum(len(label.split()) for label in cleaned_labels) / len(
            cleaned_labels
        )

        # Get top 20 most frequent words
        top_20_words = word_counts.most_common(20)

        # Get bottom 20 least frequent words
        bottom_20_words = word_counts.most_common()[:-21:-1]

        # Calculate total number of words
        total_words = len(words)

        return {
            "longest_label_length": longest_label_length,
            "shortest_label_length": shortest_label_length,
            "longest_label_length": longest_label_length,
            "shortest_label_length": shortest_label_length,
            "words": words,
            "unique_words": unique_words,
            "unique_words_count": len(unique_words),
            "top_20_words": top_20_words,
            "bottom_20_words": bottom_20_words,
            "avg_sentence_length": avg_sentence_length,
            "total_words": total_words,
        }

    def _train_test_label_compare(self):
        """
        Compares label statistics between training and testing sets.

        Returns:
            dict: Dictionary containing various statistics and comparisons.
        """
        # Split labels into training and testing sets
        train_labels = [
            label for label, flag in zip(self.labels, self.val_flag) if not flag
        ]
        test_labels = [label for label, flag in zip(self.labels, self.val_flag) if flag]

        # Get statistics for training and testing sets
        # The statistics include the number of words, unique words, and the words themselves
        # And the labels are "cleaned" so use these for further stats.
        train_stats = self._get_label_statistics(train_labels)
        test_stats = self._get_label_statistics(test_labels)

        unique_train_words = train_stats["unique_words"]
        unique_test_words = test_stats["unique_words"]

        # Calculate common, unique to training, and unique to testing words
        common_words = unique_train_words & unique_test_words
        words_unique_to_train = unique_train_words - unique_test_words
        words_unique_to_test = unique_test_words - unique_train_words

        # For the common_words, what is their frequency in the training and testing sets?
        common_word_counts_train = {
            word: train_stats["words"].count(word) for word in common_words
        }
        common_word_counts_test = {
            word: test_stats["words"].count(word) for word in common_words
        }

        # Get top 20 common words in training and testing sets
        top_20_common_words_train = collections.Counter(
            common_word_counts_train
        ).most_common(20)
        top_20_common_words_test = collections.Counter(
            common_word_counts_test
        ).most_common(20)

        # Get top 20 unique words in training and testing sets
        top_20_unique_train_words = collections.Counter(
            {word: train_stats["words"].count(word) for word in words_unique_to_train}
        ).most_common(20)
        top_20_unique_test_words = collections.Counter(
            {word: test_stats["words"].count(word) for word in words_unique_to_test}
        ).most_common(20)

        # self.label_masks is a list of boolean tensors, one per label
        # Use self.label_masks to find the longest embedded label length.
        # This is useful to help set the value of max_gen_seq_len
        # for the T5 model.  Do similar for the embedded input data via
        # attention_masks.
        max_embedded_label_len = max([mask.sum() for mask in self.label_masks])
        max_attention_mask_len = max([mask.sum() for mask in self.attention_masks])

        label_stats = {
            "train_stats": train_stats,
            "test_stats": test_stats,
            "common_words_count": len(common_words),
            "unique_train_words_count": len(unique_train_words),
            "unique_test_words_count": len(unique_test_words),
            "top_20_common_words_train": top_20_common_words_train,
            "top_20_common_words_test": top_20_common_words_test,
            "top_20_unique_train_words": top_20_unique_train_words,
            "top_20_unique_test_words": top_20_unique_test_words,
            "max_embedded_label_len": max_embedded_label_len,
            "max_attention_mask_len": max_attention_mask_len,
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
        train_stats_filtered = {
            k: v
            for k, v in label_stats["train_stats"].items()
            if k not in ["unique_words", "words"]
        }
        test_stats_filtered = {
            k: v
            for k, v in label_stats["test_stats"].items()
            if k not in ["unique_words", "words"]
        }

        print("Training Set Statistics:")
        print(
            tabulate(
                train_stats_filtered.items(),
                headers=["Statistic", "Value"],
                tablefmt="grid",
            )
        )

        print("\nTesting Set Statistics:")
        print(
            tabulate(
                test_stats_filtered.items(),
                headers=["Statistic", "Value"],
                tablefmt="grid",
            )
        )

        print("\nCommon Words Count:", label_stats["common_words_count"])
        print("Unique Train Words Count:", label_stats["unique_train_words_count"])
        print("Unique Test Words Count:", label_stats["unique_test_words_count"])

        print("\nTop 20 Common Words in Training Set:")
        print(
            tabulate(
                label_stats["top_20_common_words_train"],
                headers=["Word", "Count"],
                tablefmt="grid",
            )
        )

        print("\nTop 20 Common Words in Testing Set:")
        print(
            tabulate(
                label_stats["top_20_common_words_test"],
                headers=["Word", "Count"],
                tablefmt="grid",
            )
        )

        print("\nTop 20 Unique Train Words:")
        print(
            tabulate(
                label_stats["top_20_unique_train_words"],
                headers=["Word", "Count"],
                tablefmt="grid",
            )
        )

        print("\nTop 20 Unique Test Words:")
        print(
            tabulate(
                label_stats["top_20_unique_test_words"],
                headers=["Word", "Count"],
                tablefmt="grid",
            )
        )

        print("\nMax Embedded Label Length:", label_stats["max_embedded_label_len"])
        print("Max Attention Mask Length:", label_stats["max_attention_mask_len"])
