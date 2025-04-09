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
        tensorboard --logdir='/home/ubuntu/speechBCI/data/competitionData/tensorboard/' --port=6008
        # Then open browser to http://localhost:6006/
        
    Monitoring Learning Progres:
            Look at the predicted text and compare to the original text for training set.
        Go to llm_model_dir and look at the predictions files...
        cat MM_LLM_BART_ATTENTION_VQ_VAE_64_512_test_set_epoch_9_predictions.txt  | grep "Predicted (original)" | sort | uniq -c
            
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
from mmllm.diagnostic_utils import ModelDiagnostics

from dev_SpeechBCIDataSet_Embedded import SpeechBCIDataSet_Embedded
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
    script_name = "dev_mmllm"
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
    model_type = "bart"  # Change to 'bart' to use BART instead
    
    # Choose adapter type: 'linear', 'lstm', 'conv', 'attention', or 'rnn'
    adapter_type = "attention"  # Change this to 'rnn' to use the new RNN adapter
    num_heads = 8 # Number of attention heads for the transformer adapter
    num_layers = 2 # Number of transformer layers for the transformer adapter
    dropout = 0.1  # Dropout rate for the transformer adapter
    diversity_loss_weight = 0.1 # Additional loss term for diversity
    adapter_reg_weight = 0.05 # Additional loss term for adapter regularization

    # Add these after adapter_type
    attention_mode = "global"  # Options: 'global', 'causal', 'local' 
    window_size = None  # For local attention window size

    # Experiment name(s) (composite)
    vqvae_model_name = "VQ_VAE_64_512"
    exp_name = f"MM_LLM_{model_type.upper()}" + f"_{adapter_type.upper()}" + f"_{vqvae_model_name}"
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
    mmllm_model_dir = os.path.join(models_base_dir, exp_name) 
    os.makedirs(mmllm_model_dir, exist_ok=True)

    # TensorBoard directory
    tensorboard_dir = os.path.join(tensorboard_base_dir, exp_name)  # This will be written to
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Hyperparameters
        # Need to reference the input data and VQVAE dimensions.  Align with main_vqvae3D.py
    num_ecog_input_channels = 4
    num_encoder_out_channels = 128
    
    vqvae_embed_dim = 64  # Rename for consistency (was embedding_dim)
    vqvae_num_embeddings = 512
    
    llm_embed_dim = vqvae_embed_dim * 8 * 4  # (which we set in main_vqvae3D.py)
    
    max_input_seq_len = 256  # Padding to get batch dimension uniformity (not LLM requirements, per se).
    num_epochs = 20
    learning_rate = 1e-5
    training = True
    test_prop = 0.2
    train_prop = 1 - test_prop
    batch_size = 24
    max_gen_seq_len = 32
    num_gen_beams = 2
    eval_training_set = False
    
    # Diagnostics set up
    enable_diagnostics = True
    
    # VQVAE model for embedding preparation
    vqvae_model = VQVAE(num_ecog_input_channels, num_encoder_out_channels, vqvae_embed_dim, vqvae_num_embeddings)
    vqvae_model.load_state_dict(torch.load(os.path.join(vqvae_model_dir, vqvae_model_name + "_final.pt")))
    padding_vector = get_vqvae_codebook_average(vqvae_model)
    #print(vqvae_model)
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
        max_seq_len=max_input_seq_len,
        padding_vector=padding_vector,  # padding_vector has dim [vqvae_embed_dim]
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

    # Create MMLLM model and adapter as separate components
    mm_llm, adapter = create_embedding_model(
        model_type=model_type, 
        base_model=lora_base_model, 
        input_dim=llm_embed_dim,  # Changed from embed_dim to input_dim
        adapter_type=adapter_type,
        attention_mode=attention_mode,
        window_size=window_size,
        total_input_dim=(max_input_seq_len * llm_embed_dim) if adapter_type == "linear" else None,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    
    if enable_diagnostics:
        model_diagnostics = ModelDiagnostics(
            model=mm_llm,
            adapter=adapter,  # Add adapter to diagnostics
            tokenizer=tokenizer,
            tensorboard_dir=tensorboard_dir,
            output_dir=os.path.join(mmllm_model_dir, "diagnostics"),
        )
    else:
        model_diagnostics = None

    # Display model information
    mm_llm.print_trainable_parameters()
    # Add adapter parameter display
    if hasattr(adapter, "print_trainable_parameters"):
        adapter.print_trainable_parameters()
    else:
        adapter_params = sum(p.numel() for p in adapter.parameters())
        adapter_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        print(f"\nAdapter parameters: {adapter_params:,}")
        print(f"Adapter trainable parameters: {adapter_trainable:,} ({100*adapter_trainable/adapter_params:.2f}%)")

    # Set up optimizer with parameters from both model and adapter
    optimizer_params = list(mm_llm.parameters()) + list(adapter.parameters()) 
    optimizer = torch.optim.AdamW(optimizer_params, lr=learning_rate)

    print(f"\nExperiment Configuration:")
    print(f"- Model Type: {model_type}")
    print(f"- Adapter Type: {adapter_type}")
    if adapter_type == "attention":
        print(f"- Attention Mode: {attention_mode}")
        print(f"- Window Size: {window_size}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Batch Size: {batch_size}")
    print(f"- Max Sequence Length: {max_input_seq_len}")
    print(f"- Num Epochs: {num_epochs}")

    if training:
        # Train and evaluate the model
        trained_model = run_exp(
            exp_name=exp_name,
            train_dl=train_dl,
            test_dl=test_dl,
            val_dl=val_dl,
            model=mm_llm,
            adapter=adapter,  # Added adapter parameter
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
            diversity_loss_weight=diversity_loss_weight,
            adapter_reg_weight=adapter_reg_weight,
            diagnostics=model_diagnostics,
        )

    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"*** {script_name} - END ***")


if __name__ == "__main__":
    main()