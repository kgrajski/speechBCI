"""
Data utilities for SpeechBCI multimodal language models
"""

import torch
import os
import numpy as np
from jiwer import wer
import string
from tabulate import tabulate


def log_metrics(writer, exp_name, description, metrics, epoch):
    """Log metrics to TensorBoard and print to stdout using tabulate.

    Args:
        writer: TensorBoard writer object.
        exp_name: Experiment name.
        description: Description of the metrics (e.g., "training", "validation").
        metrics: A dictionary where keys are metric names and values are metric values.
        epoch: Epoch number.
    """
    # Log to TensorBoard
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(
            f"{exp_name}/{description}_{metric_name}", metric_value, epoch
        )
    writer.flush()

    # Print to stdout using tabulate
    table = []
    for metric_name, metric_value in metrics.items():
        table.append([metric_name, metric_value])

    headers = ["Metric", "Value"]
    print(f"\n{exp_name} - {description} - Epoch: {epoch}")
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print("\n")


# Process generated text to ensure proper sentence format
def process_generated_output(text, model_type):
    """Process a single generated text.

    Args:
        text: Decoded text string
        model_type: Type of model (t5, bart, etc.)

    Returns:
        Processed text string
    """
    # For BART models, handle sentence tokens and ensure single sentence
    if model_type.lower() == "bart":
        # Remove sentence boundary tokens
        text = text.replace("<sentence>", "").replace("</sentence>", "").strip()

        # Ensure we have a single sentence with proper ending
        if "." in text:
            # Take first sentence and ensure it ends with period
            text = text.split(".")[0].strip() + "."
        elif len(text) > 0 and not text.endswith("."):
            # Add period if missing
            text = text.strip() + "."

    return text


def process_generated_texts(texts, model_type, task_prompt=""):
    """Process a list of generated texts.

    Args:
        texts: List of decoded text strings
        model_type: Type of model (t5, bart, etc.)
        task_prompt: The prompt that was used for generation (e.g., "Generate a sentence:")

    Returns:
        List of processed text strings
    """
    processed_texts = []
    for text in texts:
        # Remove the task prompt
        if task_prompt and text.startswith(task_prompt):
            text = text[len(task_prompt):].strip()

        # For BART models, handle sentence tokens and ensure single sentence
        if model_type.lower() == "bart":
            # Remove sentence boundary tokens
            text = text.replace("<sentence>", "").replace("</sentence>", "").strip()

            # Ensure we have a single sentence with proper ending
            if not (text.endswith(".") or text.endswith("?") or text.endswith("!")):
                text += "."

        processed_texts.append(text)
    return processed_texts


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
