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
    - python dev_train_vqvae.py > training_log.txt 2>&1
    - Press Ctrl+A, then press D to detach from the screen.
    - screen -ls
    - screen -r speechBCI_training
    - screen -X -S speechBCI_training quit

"""

import gc
import numpy as np
import os
import time
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from utils_train_vqvae_ray import train_vqvae, setup_data_loaders


def main():
    """
    Main function to perform hyperparameter search for VQ-VAE training using ray hyperparameter search.
    """
    script_name = "dev_train_vqvae"
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
    # Directories
    root_dir = "/home/ubuntu"
    project_dir = os.path.join(root_dir, "speechBCI")
    data_dir = os.path.join(project_dir, "data/competitionData")
    model_dir = os.path.join(data_dir, "models")
    tensorboard_dir = os.path.join(data_dir, "tensorboard")
    
    ray_log_dir = os.path.join(data_dir, "ray")
    os.makedirs(ray_log_dir, exist_ok=True)

    # Hyperparameter search space
    search_space = {
        "embedding_dim": tune.grid_search([64, 128, 256, 512]),
        "num_embeddings": tune.grid_search([64, 128, 256, 512]),
        "num_ecog_input_channels": 4,
        "num_encoder_out_channels": 128,
        "learning_rate": 1e-3,
        "num_epochs": 25,
    }

    # Dataset setup (done once)
    # When usng ray, it is better to not pass the data objects directly to the function
    batch_size = 64
    encoder_depth = 8
    depth_step_size = 4

    # Metrics setup
    metric = "reconstruction_loss_train"
    mode = "min"
    
    # Dynamically set max_t to match the maximum value of num_epochs in the search space
    scheduler = ASHAScheduler(max_t=100, grace_period=10, reduction_factor=2)

    # Run hyperparameter search
    print("Starting hyperparameter search...")
    print(f"Search space: {search_space}")
    analysis = tune.run(
        tune.with_parameters(
            train_vqvae,
            model_dir=model_dir,
            tensorboard_dir=tensorboard_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            data_dir=data_dir,               # Pass paths instead of data objects
            batch_size=batch_size,
            encoder_depth=encoder_depth,
            depth_step_size=depth_step_size,
        ),
        resources_per_trial={"cpu": 4, "gpu": 0.98},  # Resources allocated per trial
        config=search_space,  # Hyperparameter search space
        scheduler=scheduler,  # Scheduler for managing trials
        metric=metric,
        mode=mode,
        num_samples=1,  # Ensure each combination is evaluated only once
        storage_path=ray_log_dir,  # Redirect logs to this directory
    )
    
    print("Best hyperparameters found were: ", analysis.best_config)
    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")

if __name__ == "__main__":
    main()