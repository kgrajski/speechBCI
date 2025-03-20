"""
Multimodal Language Model for Speech BCI Data

This module fine-tunes a T5 language model with LoRA adaptation to process
Speech BCI embeddings and generate text. It loads pre-trained VQVAE embeddings
and uses a custom adapter architecture for multimodal integration.

Functions:
    main(): Sets up and runs the multimodal language model training experiment.
"""

"""
Development Reminders:
    
    GPU Monitoring:
        nvidia-smi -l 5  # Updates every 5 seconds
        nvidia-smi --id=0 --loop=30 --query --display=UTILIZATION
    
    TensorBoard Visualization:
        tensorboard --logdir='/home/ubuntu/speechBCI/data/competitionData/tensorboard/MM_LLM'
        # Then open browser to http://localhost:6006/
        
    Monitoring Learning Progres:
    Go to llm_model_dir and look at the predictions files...
    cat MM_LLM_training_set_epoch_3_predictions.txt  | grep "Predicted" | sort | uniq -c
    
"""

#
# Updated: March 19, 2025
#

import sys

sys.path.append("./")

import gc
import numpy as np
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

from SpeechBCIDataSet_Embedded import SpeechBCIDataSet_Embedded
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils_mmllm import (
    get_vqvae_codebook_average,
    run_exp,
    CustomEmbeddingT5,
    get_lora_model,
)
from Vqvae_Simple3D import VQVAE


def main():
    """
    Main function to set up the experiment and run it.
    """
    script_name = "main_mmllm"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device=", device)
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    exp_name = "MM_LLM"
    model_exp_name = "VQVAE_Simple_3D"

    etl_dir = "/home/ubuntu/speechBCI/data/competitionData/etl"
    embed_dir = "/home/ubuntu/speechBCI/data/competitionData/embeddings"

    model_dir = "/home/ubuntu/speechBCI/data/competitionData/models"
    model_dir = os.path.join(model_dir, model_exp_name)

    mmllm_model_dir = "/home/ubuntu/speechBCI/data/competitionData/models"
    mmllm_model_dir = os.path.join(mmllm_model_dir, exp_name)
    os.makedirs(mmllm_model_dir, exist_ok=True)

    tensorboard_dir = "/home/ubuntu/speechBCI/data/competitionData/tensorboard"
    tensorboard_dir = os.path.join(tensorboard_dir, exp_name)
    os.makedirs(tensorboard_dir, exist_ok=True)

    embedding_dim = 64  # Later can make this automatic.
    max_seq_len = 512
    num_epochs = 100
    learning_rate = 1e-4
    training = True

    test_prop = 0.2
    train_prop = 1 - test_prop
    batch_size = 16
    max_gen_seq_len = 64
    num_gen_beams = 3

    #
    # Need model info to set up the optimizer and prep for transformer, such as
    # by computing the average codebook vector.  May be model-dependent.
    #
    model = VQVAE()
    model.load_state_dict(
        torch.load(os.path.join(model_dir, model_exp_name + "_final" + ".pt"))
    )
    padding_vector = get_vqvae_codebook_average(model)
    del model  # Don't need the model after this point.

    #
    # Per Willett, et al. competition data, the last block in each session
    # should be used as the test set.  Here, we'll call that set the validation set
    # and split the remaining data into training and validation sets.
    # Note: in the official competition, there is a distinct validation (holdout) set.
    # In MVP stage, there will be model-dependent methods in SpeechBCIDataSet_Embedded
    #
    # torch.autograd.set_detect_anomaly(True)
    study_dataset = SpeechBCIDataSet_Embedded(
        etl_dir, embed_dir, max_seq_len, padding_vector
    )

    #
    # Recall that we are using competition data.  That study defines the last block in
    # each session as the test set.  And in such case the validation set is the data
    # that was withheld. At the risk of short changing the training set here, we'll
    # use the last block in each session as the withheld validation set.  The remaining
    # data we'll split into the traditional training and test set.
    # Consequently, it makes sense to have a quick
    #
    # Generate some statistics on the raw words in the training and testing sets.
    #
    study_dataset._train_test_label_compare()

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

    if training:
        # Load the base T5 model
        t5_model_name = "t5-small"
        t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

        # Apply LoRA to T5
        lora_model = get_lora_model(
            t5_model,
            r=8,  # LoRA rank - smaller = fewer parameters
            alpha=32,  # LoRA alpha scaling factor
            dropout=0.1,  # LoRA dropout rate
        )

        # Create custom embedding adapter with LoRA model
        mm_llm = CustomEmbeddingT5(lora_model, embedding_dim=embedding_dim)

        # Print parameter efficiency statistics
        mm_llm.print_trainable_parameters()

        tokenizer = T5Tokenizer.from_pretrained(t5_model_name, legacy=True)

        # Create optimizer - only trainable parameters will be updated
        optimizer = torch.optim.AdamW(
            mm_llm.parameters(), lr=learning_rate
        )  # Using a higher LR for LoRA

        # Train and evaluate model
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
        )

        print(trained_model)

    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")


if __name__ == "__main__":
    main()
