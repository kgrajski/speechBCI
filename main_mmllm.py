"""
Multimodal Language Model for Speech BCI Data

This module fine-tunes a language model with LoRA adaptation to process
Speech BCI embeddings and generate text. It supports multiple model types
including T5 and BART.

Functions:
    main(): Sets up and runs the multimodal language model training experiment.
"""

"""
Development Reminders:
    
    GPU Monitoring:
        nvidia-smi -l 5  # Updates every 5 seconds
        nvidia-smi --id=0 --loop=30 --query --display=UTILIZATION
    
    TensorBoard Visualization:
        tensorboard --logdir='/home/ubuntu/speechBCI/data/competitionData/tensorboard/MM_LLM_T5_VQVAE_4C_16H_8W_128_256_r2/' --port=6008
        # Then open browser to http://localhost:6006/
        
    Monitoring Learning Progres:
            Look at the predicted text and compare to the original text for training set.
        Go to llm_model_dir and look at the predictions files...
        cat MM_LLM_T5_training_set_epoch_7_predictions.txt  | grep "Predicted (original)" | sort | uniq -c
            
            Look at the number of unique words being predicted.
        cat MM_LLM_BART_training_set_epoch_4_predictions.txt | grep "Predict" | grep "original" | sort | uniq -c | \
            awk -F':' '{print $2}' | tr ' ' '\n'| sort | uniq | wc
        cat MM_LLM_BART_training_set_epoch_4_predictions.txt | grep "Predict" | grep "standard" | sort | uniq -c | \
            awk -F':' '{print $2}' | tr ' ' '\n'| sort | uniq | wc
            
    Screen (in an ssh command line session; haven't tried from VSS terminal)
    - screen -S speechBCI_training
    - cd /home/ubuntu/speechBCI
    - source .venv/bin/activate
    - python main_mmllm.py > training_log.txt 2>&1
    - Press Ctrl+A, then press D to detach from the screen.
    - screen -ls
    - screen -r speechBCI_training
    - screen -X -S speechBCI_training quit

"""

import sys
import os
import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset  # Added Subset

from SpeechBCIDataSet_Embedded import SpeechBCIDataSet_Embedded
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
)

from mmllm.data_utils import get_vqvae_codebook_average
from mmllm.model_utils import create_embedding_model, get_lora_model
from mmllm.label_utils import LabelAnalyzer
from mmllm.training_utils import run_exp

from Vqvae_Simple3D import VQVAE


def main():
    """
    Main function to set up the experiment and run it.
    """
    script_name = "main_mmllm"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***\n")

    # Set device and seed for reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device=", device)
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    # ECOG subset
    ecog_subset = "6v_all"  # Requirement
    
    # Choose model type: 't5' or 'bart'
    model_type = "t5"  # Change to 'bart' to use BART instead

    # Experiment name (composite)
    vqvae_model_name = "VQVAE_4C_16H_8W_128_256"
    exp_name = f"MM_LLM_{model_type.upper()}" + f"_{vqvae_model_name}" + f"_r2"
    print(f"Experiment name: {exp_name}")
  
    # Directory setup
    root_dir = "/home/ubuntu"
    project_dir = os.path.join(root_dir, "speechBCI")
    data_dir = os.path.join(project_dir, "data/competitionData")

    # Define all data directories using the common root
    etl_dir = os.path.join(data_dir, "etl", ecog_subset) # This will be read
    embed_dir = os.path.join(data_dir, "embeddings")  # This will be written to
    models_base_dir = os.path.join(data_dir, "models")
    tensorboard_base_dir = os.path.join(data_dir, "tensorboard")

    # Model-specific directories
    vqvae_model_dir = os.path.join(models_base_dir, vqvae_model_name)  # This will be read only
    embed_dir = os.path.join(embed_dir, vqvae_model_name)  # This will be read only
    mmllm_model_dir = os.path.join(models_base_dir, exp_name)  # This will be written to
    os.makedirs(mmllm_model_dir, exist_ok=True)

    # TensorBoard directory
    tensorboard_dir = os.path.join(tensorboard_base_dir, exp_name)  # This will be written to
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Hyperparameters
        # Need to reference the input data and VQVAE dimensions.  Align with main_vqvae3D.py
    num_ecog_input_channels = 4
    num_encoder_out_channels = 256
    embedding_dim = 128
    num_embeddings = 256
    
    max_seq_len = 512  # Padding to get batch dimension uniformity (not LLM requirements, per se).
    num_epochs = 1000
    learning_rate = 1e-5
    training = True
    test_prop = 0.2
    train_prop = 1 - test_prop
    batch_size = 16
    max_gen_seq_len = 64
    num_gen_beams = 3
    eval_training_set = True

    # VQVAE model for embedding preparation
    vqvae_model = VQVAE(num_ecog_input_channels, num_encoder_out_channels, embedding_dim, num_embeddings)
    vqvae_model.load_state_dict(torch.load(os.path.join(vqvae_model_dir, vqvae_model_name + "_final.pt")))
    padding_vector = get_vqvae_codebook_average(vqvae_model)
    print(vqvae_model)
    del vqvae_model

    # Load appropriate model and tokenizer based on model type
    if model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
        base_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    elif model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        # Add sentence boundary tokens
        special_tokens = {"additional_special_tokens": ["<sentence>", "</sentence>"]}
        tokenizer.add_special_tokens(special_tokens)
        base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        # Resize model embeddings to match updated tokenizer
        base_model.resize_token_embeddings(len(tokenizer))
        print("Using standard BART without multilingual support")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    #
    # Per Willett, et al. competition data, the last block in each session
    # should be used as the test set.  Here, we'll call that set the validation set
    # and split the remaining data into training and validation sets.
    # Note: in the official competition, there is a distinct validation (holdout) set.
    # In MVP stage, there will be model-dependent methods in SpeechBCIDataSet_Embedded
    # to handle this.
    # Create dataset with proper padding vector
    study_dataset = SpeechBCIDataSet_Embedded(
        embed_dir=embed_dir,
        etl_dir=etl_dir,
        tokenizer=tokenizer,
        model_type=model_type,
        max_seq_len=max_seq_len,
        padding_vector=padding_vector,
    )
    

    #
    # Recall that we are using competition data.  That study defines the last block in
    # each session as the test set.  And in such case the validation set is the data
    # that was withheld. At the risk of short changing the training set here, we'll
    # use the last block in each session as the withheld validation set.  The remaining
    # data we'll split into the traditional training and test set.
    # Consequently, it makes sense to have a quick
    
    # Check the label statistics
    label_stats = LabelAnalyzer(
        study_dataset.labels,
        study_dataset.val_flag,
        study_dataset.label_masks,
        study_dataset.attention_masks
    )
    label_stats.print_overall_stats()
    label_stats.print_train_test_comparison()

    # Now subset the study data as described above.
    train_test_indices = [
        i
        for i in range(len(study_dataset.val_flag))
        if study_dataset.val_flag[i] is False
    ]
    val_indices = [
        i
        for i in range(len(study_dataset.val_flag))
        if study_dataset.val_flag[i] is True
    ]
    train_test_dataset = Subset(study_dataset, train_test_indices)
    train_dataset, test_dataset = random_split(
        train_test_dataset, [train_prop, test_prop]
    )
    val_dataset = Subset(study_dataset, val_indices)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create LoRA model
    lora_base_model = get_lora_model(base_model, model_type=model_type)

    # Create MMLLM model with appropriate adapter
    mm_llm = create_embedding_model(
        model_type=model_type, base_model=lora_base_model, embedding_dim=embedding_dim
    )

    # Display model information
    mm_llm.print_trainable_parameters()

    # Set up optimizer
    optimizer = torch.optim.AdamW(mm_llm.parameters(), lr=learning_rate)

    if training:
        # Train and evaluate the model
        trained_model = run_exp(
            exp_name=exp_name,
            train_dl=train_dl,
            test_dl=test_dl,
            val_dl=val_dl,
            model=mm_llm,
            optimizer=optimizer,
            tokenizer=tokenizer,
            device=device,
            num_epochs=num_epochs,
            max_gen_seq_len=max_gen_seq_len,
            num_gen_beams=num_gen_beams,
            model_dir=mmllm_model_dir,
            tensorboard_dir=tensorboard_dir,
            model_type=model_type,  # Pass model type to functions
            eval_training_set=eval_training_set,
        )

    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"*** {script_name} - END ***")


if __name__ == "__main__":
    main()