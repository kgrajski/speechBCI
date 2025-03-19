import os
import re
import string
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from transformers import BartTokenizer, BartForConditionalGeneration
from SpeechBCIDataSet_Embedded_BART import SpeechBCIDataSet_Embedded_BART
from utils_mmllm_bart import CustomEmbeddingBART, get_vqvae_codebook_average, get_lora_model, run_exp
from Vqvae_Simple3D import VQVAE

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("Device=", device)
    if device == "cuda":
        torch.cuda.empty_cache()

    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    
    exp_name = "MM_BART"
    model_exp_name = "VQVAE_Simple_3D"
    
    etl_dir = "/home/ubuntu/speechBCI/data/competitionData/etl"
    embed_dir = "/home/ubuntu/speechBCI/data/competitionData/embeddings"
    
    model_dir = "/home/ubuntu/speechBCI/data/competitionData/models"
    model_dir = os.path.join(model_dir, model_exp_name)
    
    mmllm_model_dir = "/home/ubuntu/speechBCI/data/competitionData/models"
    mmllm_model_dir = os.path.join(model_dir, exp_name)
    os.makedirs(mmllm_model_dir, exist_ok=True)
    
    tensorboard_dir = "/home/ubuntu/speechBCI/data/competitionData/tensorboard"
    tensorboard_dir = os.path.join(tensorboard_dir, exp_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    embedding_dim = 64  # Later can make this automatic
    max_seq_len = 512 
    num_epochs = 5
    learning_rate = 5e-4
    training = True
    
    test_prop = 0.2
    train_prop = 1 - test_prop
    batch_size = 4
    max_gen_seq_len = 32
    num_gen_beams = 5
    
    # Load VQVAE model to get padding vector
    model = VQVAE()
    model.load_state_dict(torch.load(os.path.join(model_dir, model_exp_name + "_final" + ".pt")))
    padding_vector = get_vqvae_codebook_average(model)
    del model  # Don't need the model after this point
    
    # Load dataset with BART tokenizer
    study_dataset = SpeechBCIDataSet_Embedded_BART(etl_dir, embed_dir, max_seq_len, padding_vector)
    
    # Generate statistics on raw words in training and testing sets
    study_dataset._train_test_label_compare()
    
    # Subset the study data
    train_test_indices = [i for i in range(len(study_dataset.val_flag)) if study_dataset.val_flag[i] is False]
    val_indices = [i for i in range(len(study_dataset.val_flag)) if study_dataset.val_flag[i] is True]
    train_test_dataset = Subset(study_dataset, train_test_indices)
    train_dataset, test_dataset = random_split(train_test_dataset, [train_prop, test_prop])
    val_dataset = Subset(study_dataset, val_indices)
    
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if training:
        # Load the base BART model
        bart_model_name = "facebook/bart-base"
        bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
        bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
        
        # Apply LoRA to BART
        lora_model = get_lora_model(
            bart_model, 
            r=8,               # LoRA rank - smaller = fewer parameters
            alpha=32,          # LoRA alpha scaling factor
            dropout=0.1        # LoRA dropout rate
        )
        
        # Create custom embedding adapter with LoRA model
        mm_llm = CustomEmbeddingBART(lora_model, embedding_dim=embedding_dim)
        mm_llm.print_trainable_parameters()
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(mm_llm.parameters(), lr=learning_rate)
        
        # Run experiment
        trained_model = run_exp(
            exp_name=exp_name,
            train_dl=train_dl,
            test_dl=test_dl,
            val_dl=val_dl,
            model=mm_llm,
            optimizer=optimizer,
            tokenizer=bart_tokenizer,
            device=device,
            num_epochs=num_epochs,
            max_gen_seq_len=max_gen_seq_len,
            num_gen_beams=num_gen_beams,
            model_dir=mmllm_model_dir,
            tensorboard_dir=tensorboard_dir
        )
    else:
        # Logic for loading pretrained model would go here
        print("Loading pretrained model not implemented yet")

if __name__ == "__main__":
    main()