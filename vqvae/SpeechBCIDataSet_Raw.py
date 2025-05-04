"""
Raw dataset class for Speech BCI data.

This module provides a PyTorch dataset that loads VQ-VAE embeddings from speech BCI data
without preprocessing, allowing more flexibility in handling temporal sequences.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from etl.Sentence import Sentence


class SpeechBCIDataSet_Raw(Dataset):
    """
    Dataset for raw VQ-VAE embeddings from Speech BCI data.

    Unlike SpeechBCIDataSet_Embedded, this class loads embeddings without padding
    or masking, preserving the original temporal sequence for hierarchical compression.
    """

    def __init__(self, embed_dir, etl_dir):
        """
        Initialize the raw dataset.

        Args:
            embed_dir: Directory containing VQ-VAE embeddings
            etl_dir: Directory containing label data
        """
        self.embed_dir = embed_dir
        self.etl_dir = etl_dir

        # Initialize storage
        self.trial_ids = []  # Trial identifiers
        self.labels = []  # Text labels
        self.val_flag = []  # Validation flags

        # Load trial information
        self._load_trials()

        print(f"Loaded {len(self.trial_ids)} trials")
        print(f"- Training: {len([f for f in self.val_flag if not f])}")
        print(f"- Validation: {len([f for f in self.val_flag if f])}")

    def _load_trials(self):
        """Load trial information and labels."""
        for split in ["train", "test"]:
            # Get list of trials in this split
            trial_dir = os.path.join(self.embed_dir, split)
            if not os.path.exists(trial_dir):
                raise ValueError(f"Directory not found: {trial_dir}")

            # List all trial files (*.pt files)
            trial_files = [f for f in os.listdir(trial_dir) if f.endswith(".pt")]

            for file in trial_files:
                # Extract trial ID (remove extension)
                trial_id = file.split(".")[0]

                # Construct path to label file
                label_file = os.path.join(self.etl_dir, f"{trial_id}_sentenceText.csv")

                # Load label text if available
                if os.path.exists(label_file):
                    text_data = Sentence()
                    text_data.load(label_file)
                    label_text = text_data.sentence_txt

                    # Store trial information
                    self.trial_ids.append((split, trial_id))
                    self.labels.append(label_text)
                    self.val_flag.append(split == "test")

    def __len__(self):
        """Return the number of trials in the dataset."""
        return len(self.trial_ids)

    def __getitem__(self, idx):
        """
        Get raw embeddings and label for a trial.

        Args:
            idx: Index of the trial

        Returns:
            Dictionary containing raw embeddings and label
        """
        # Get trial information
        split, trial_id = self.trial_ids[idx]

        # Load embeddings
        file_path = os.path.join(self.embed_dir, split, f"{trial_id}.pt")
        embeddings = torch.load(file_path)

        # Return raw data
        return {
            "trial_id": trial_id,
            "vqvae_embeddings": embeddings,  # Shape: [seq_len, embedding_dim]
            "label": self.labels[idx],
            "is_validation": self.val_flag[idx],
        }

    def get_spatial_shape(self):
        """
        Determine the spatial shape of the VQ-VAE embeddings.

        Returns:
            Tuple (H, W) representing the spatial dimensions
        """
        # Load first trial to analyze shape
        if len(self.trial_ids) == 0:
            return None

        split, trial_id = self.trial_ids[0]
        file_path = os.path.join(self.embed_dir, split, f"{trial_id}.pt")
        sample = torch.load(file_path)

        # Analyze shape to determine spatial dimensions
        # This depends on how the VQ-VAE embeddings are stored
        # For a typical 8×4 grid, each time step should have 32 tokens

        # Check if sample is already reshaped correctly
        if len(sample.shape) == 3:  # [time, spatial, dim]
            return (sample.shape[1], 1)  # Assuming [time, spatial, dim]

        # Otherwise, try to infer from typical configurations
        # Common configurations: 8×4=32, 4×4=16, etc.
        if len(sample.shape) == 2:  # [time*spatial, dim]
            # Analyze embedding lengths and patterns
            # This is a heuristic and might need adaptation
            seq_len = sample.shape[0]

            # Try common spatial shapes
            for h, w in [(8, 4), (4, 4), (4, 2)]:
                spatial_tokens = h * w
                if seq_len % spatial_tokens == 0:
                    return (h, w)

        # Default to 8×4 if unable to determine
        print("Warning: Could not determine spatial shape, using default (8, 4)")
        return (8, 4)
