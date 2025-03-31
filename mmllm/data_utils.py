"""
Data utilities for SpeechBCI multimodal language models
"""

import torch
import os
import numpy as np
from jiwer import wer
import string


def get_vqvae_codebook_average(model):
    """
    Calculate the average of all embedding vectors in the VQ-VAE codebook.

    This function extracts the codebook embeddings from a trained VQ-VAE model
    and computes their mean. This can be useful for padding or initialization
    purposes when using the codebook representations.

    Args:
        model (VQVAE): A trained VQ-VAE model instance

    Returns:
        torch.Tensor: The average embedding vector with shape [embedding_dim]
    """
    # Get the vector quantizer (either standard or EMA)
    vq = model._vq_vae

    # Extract the embedding weights (codebook vectors)
    # Shape: [num_embeddings, embedding_dim]
    codebook = vq._embedding.weight.data

    # Calculate the average across all codebook vectors
    # Shape: [embedding_dim]
    avg_vector = torch.mean(codebook, dim=0)

    return avg_vector


def calculate_wer(predictions, references, tokenizer, model_dir, split_name, already_decoded=False):
    """Calculate WER between predictions and references.
    
    Args:
        predictions: List of generated text or token IDs
        references: List of reference texts
        tokenizer: Tokenizer for any additional processing
        model_dir: Directory to save prediction reports
        name_prefix: Prefix for the report filename
        already_decoded: Whether predictions are already decoded text (True) or token IDs (False)
        
    Returns:
        tuple: (standardized_wer, original_wer, unique_word_count)
    """
    # Decode predictions if needed
    if already_decoded:
        decoded_preds = predictions
    else:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Standardize both predictions and original texts
    standardized_preds = [standardize_text(pred) for pred in decoded_preds]
    standardized_targets = [standardize_text(text) for text in references]

    # Save both original and standardized versions to file
    fname = os.path.join(model_dir, f"{split_name}_predictions.txt")
    print(f"calculate_wer {split_name} wrote file {fname}")
    with open(fname, "w") as f:
        for orig_target, orig_pred, std_target, std_pred in zip(
            references, decoded_preds, standardized_targets, standardized_preds
        ):
            f.write(f"Target (original): {orig_target}\n")
            f.write(f"Target (standardized): {std_target}\n")
            f.write(f"Predicted (original): {orig_pred}\n")
            f.write(f"Predicted (standardized): {std_pred}\n")
            f.write("\n")

    # Calculate WER using standardized text
    std_wer_scores = [
        wer(std_target, std_pred)
        for std_target, std_pred in zip(standardized_targets, standardized_preds)
    ]
    std_mean_wer = np.mean(std_wer_scores)

    # Calculate WER using original text
    orig_wer_scores = [
        wer(orig_target, orig_pred)
        for orig_target, orig_pred in zip(references, decoded_preds)
    ]
    orig_mean_wer = np.mean(orig_wer_scores)
    
    # The variable decoded_preds is a list of a list of strings. Using list
    # comprehension, find the number of unique words and return as num_uniq_pred_words
    num_uniq_pred_words = len(set([word for sent in decoded_preds for word in sent.split()]))
    
    return std_mean_wer, orig_mean_wer, num_uniq_pred_words


def standardize_text(text):
    """
    Standardize text for consistent comparison.

    Args:
        text: Input text string

    Returns:
        Standardized text string
    """
    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = " ".join(text.split())

    # Handle punctuation (option 1: remove punctuation)
    # text = text.translate(str.maketrans("", "", string.punctuation))

    # Handle punctuation (option 2: standardize with spaces)
    for punct in string.punctuation:
        text = text.replace(punct, f" {punct} ")
    text = " ".join(text.split())

    # Handle numbers (optional - expand if needed)
    # Can add number normalization if needed

    return text
