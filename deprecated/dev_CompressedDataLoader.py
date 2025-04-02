"""
Compressed data module for neural BCI data.

This module provides dataset and dataloader wrappers that integrate
hierarchical compression with language model training for the Speech BCI system.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import numpy as np


class CompressedDataLoader:
    """
    DataLoader wrapper that applies hierarchical compression on-the-fly.
    
    This class wraps a standard PyTorch DataLoader and processes batches
    through a hierarchical compressor before returning them.
    """
    
    def __init__(
        self,
        dataset,
        compressor,
        tokenizer,
        model_type,
        max_seq_len=512,
        batch_size=16,
        shuffle=False,
        device="cuda",
        num_workers=0
    ):
        """
        Initialize the compressed dataloader.
        
        Args:
            dataset: Dataset instance (SpeechBCIDataSet_Raw)
            compressor: HierarchicalAttentionCompressor instance
            tokenizer: Tokenizer for processing text labels
            model_type: Type of language model ('t5' or 'bart')
            max_seq_len: Maximum sequence length for labels
            batch_size: Batch size for data loading
            shuffle: Whether to shuffle the dataset
            device: Device to use for compression
            num_workers: Number of worker processes for data loading
        """
        self.dataset = dataset
        self.compressor = compressor.to(device)
        self.tokenizer = tokenizer
        self.model_type = model_type.lower()
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device
        
        # Set compressor to evaluation mode during data loading
        self.compressor.eval()
        
        # Create base dataloader with collate_fn=None to get list of samples
        self.base_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: x  # Return list of samples without batching
        )
    
    def __iter__(self):
        """Iterate through the dataloader, processing batches on-the-fly."""
        for samples in self.base_dataloader:
            # Process batch with compressor
            processed_batch = self._process_batch(samples)
            yield processed_batch
    
    def __len__(self):
        """Return the number of batches."""
        return len(self.base_dataloader)
    
    def _process_batch(self, samples):
        """
        Process a batch of samples through the compressor.
        
        Args:
            samples: List of dictionaries from dataset.__getitem__
            
        Returns:
            Dictionary with processed batch data
        """
        # Extract embeddings and labels
        embeddings = [sample["vqvae_embeddings"] for sample in samples]
        labels = [sample["label"] for sample in samples]
        trial_ids = [sample["trial_id"] for sample in samples]
        
        # Process embeddings with compressor
        compressed_data = self._compress_embeddings(embeddings)
        
        # Process labels
        processed_labels = self._process_labels(labels)
        
        # Combine into batch dictionary
        batch = {
            "vqvae_embeddings": compressed_data["embeddings"],
            "attention_mask": compressed_data["attention_mask"],
            "label_embeddings": processed_labels["ids"],
            "label_attention_mask": processed_labels["attention_mask"],
            "original_labels": labels,
            "trial_ids": trial_ids
        }
        
        return batch
    
    def _compress_embeddings(self, embeddings_list):
        """
        Compress a list of embedding tensors.
        
        Args:
            embeddings_list: List of tensors with VQ-VAE embeddings
            
        Returns:
            Dictionary with compressed embeddings and attention mask
        """
        # Move compressor to device and set to eval mode
        self.compressor.eval()
        
        with torch.no_grad():
            # Prepare and stack embeddings for batch processing
            # First, detect the embedding dimension
            emb_dim = embeddings_list[0].size(-1)
            
            # Process each sequence separately (variable length)
            compressed_embeddings = []
            
            for emb in embeddings_list:
                # Move to device
                emb_tensor = emb.to(self.device)
                
                # Add batch dimension
                emb_tensor = emb_tensor.unsqueeze(0)
                
                # Apply compression
                compressed = self.compressor(emb_tensor)
                
                # Remove batch dimension
                compressed = compressed.squeeze(0).cpu()
                compressed_embeddings.append(compressed)
            
            # Stack embeddings (all should now have same length)
            stacked_embeddings = torch.stack(compressed_embeddings)
            
            # Create attention mask (all 1s since all tokens are valid)
            attention_mask = torch.ones(
                stacked_embeddings.size(0),
                stacked_embeddings.size(1),
                dtype=torch.bool
            )
        
        return {
            "embeddings": stacked_embeddings,
            "attention_mask": attention_mask
        }
    
    def _process_labels(self, labels):
        """
        Process text labels for the language model.
        
        Args:
            labels: List of text labels
            
        Returns:
            Dictionary with processed labels
        """
        # Apply special tokens for BART
        if self.model_type == "bart":
            processed_labels = [f"<sentence> {label} </sentence>" for label in labels]
        else:
            processed_labels = labels
        
        # Tokenize with appropriate parameters
        encoded_labels = self.tokenizer(
            processed_labels,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )
        
        # Create label embeddings
        input_ids = encoded_labels.input_ids
        
        # Handle padding token IDs based on model type
        if self.model_type == "t5":
            # For T5, replace pad tokens with -100 (ignored in loss)
            input_ids_copy = input_ids.clone()
            input_ids_copy[input_ids == self.tokenizer.pad_token_id] = -100
            input_ids = input_ids_copy
        
        return {
            "ids": input_ids,
            "attention_mask": encoded_labels.attention_mask
        }


def create_data_loaders(
    raw_dataset,
    compressor,
    tokenizer,
    model_type,
    batch_size=16,
    max_seq_len=512,
    test_prop=0.2,
    device="cuda",
    seed=42
):
    """
    Create train, test, and validation data loaders with compression.
    
    Args:
        raw_dataset: SpeechBCIDataSet_Raw instance
        compressor: HierarchicalAttentionCompressor instance
        tokenizer: Tokenizer for processing labels
        model_type: Type of language model ('t5' or 'bart')
        batch_size: Batch size for data loading
        max_seq_len: Maximum sequence length for labels
        test_prop: Proportion of non-validation data to use for testing
        device: Device to use for compression
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, test_loader, val_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Split dataset into train/test and validation
    train_test_indices = [i for i in range(len(raw_dataset)) 
                         if not raw_dataset.val_flag[i]]
    val_indices = [i for i in range(len(raw_dataset)) 
                  if raw_dataset.val_flag[i]]
    
    # Create subsets
    train_test_set = Subset(raw_dataset, train_test_indices)
    val_set = Subset(raw_dataset, val_indices)
    
    # Split train_test into train and test
    train_size = int((1 - test_prop) * len(train_test_set))
    test_size = len(train_test_set) - train_size
    
    train_set, test_set = random_split(
        train_test_set, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders with compression
    train_loader = CompressedDataLoader(
        train_set,
        compressor,
        tokenizer,
        model_type,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=True,
        device=device
    )
    
    test_loader = CompressedDataLoader(
        test_set,
        compressor,
        tokenizer,
        model_type,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=False,
        device=device
    )
    
    val_loader = CompressedDataLoader(
        val_set,
        compressor,
        tokenizer,
        model_type,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=False,
        device=device
    )
    
    return train_loader, test_loader, val_loader