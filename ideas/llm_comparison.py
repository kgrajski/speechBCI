"""
This module provides tools for comparing different LLMs for the multimodal speech BCI task.
It helps evaluate which model architecture might be most suitable for your specific needs.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import json
from datetime import datetime

# Import from existing modules
from train_multimodal import get_multimodal_llm, evaluate_on_validation, train_multimodal_llm

class LLMComparator:
    """
    A utility class to compare different LLM architectures for the SpeechBCI project.
    Allows for systematic comparison of different model types, sizes, and hyperparameters.
    """
    def __init__(self, dataset, results_dir="comparison_results"):
        """
        Initialize the comparator with a dataset and results directory
        
        Args:
            dataset: SpeechBCIDataSet_Embedded instance to use for experiments
            results_dir: Directory to save comparison results
        """
        self.dataset = dataset
        self.results_dir = results_dir
        self.results = defaultdict(dict)
        self.trained_models = {}
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
    def run_comparison(self, 
                       model_configs, 
                       input_dim=512,
                       train_ratio=0.8,
                       batch_size=32,
                       epochs=5,
                       lr=5e-5,
                       save_models=True,
                       random_seed=42):
        """
        Compare multiple LLM configurations on the same dataset
        
        Args:
            model_configs: List of dicts with keys 'name', 'type', 'size'
                Example: [{'name': 't5_small', 'type': 't5', 'size': 'small'}, 
                          {'name': 'bart_base', 'type': 'bart', 'size': 'base'}]
            input_dim: Dimension of the VQ-VAE codebook vectors
            train_ratio: Ratio of data to use for training vs validation
            batch_size: Batch size for training
            epochs: Number of training epochs
            lr: Learning rate
            save_models: Whether to save trained models
            random_seed: Random seed for reproducibility
        
        Returns:
            DataFrame with comparison results
        """
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Track start time for the entire comparison
        start_time = time.time()
        
        # Run experiments for each model configuration
        for config in model_configs:
            model_name = config['name']
            model_type = config['type']
            model_size = config['size']
            
            print(f"\n{'='*50}")
            print(f"Training model: {model_name} ({model_type}-{model_size})")
            print(f"{'='*50}\n")
            
            # Track time for this specific model
            model_start_time = time.time()
            
            # Train the model
            model_save_path = os.path.join(self.results_dir, model_name) if save_models else None
            
            try:
                model, test_metrics = train_multimodal_llm(
                    dataset=self.dataset,
                    input_dim=input_dim,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=lr,
                    model_name=f"{model_type}-{model_size}" if model_type == "t5" else None,
                    save_path=model_save_path
                )
                
                # If model_type is not t5, use the factory function
                if model_type != "t5":
                    model = get_multimodal_llm(
                        model_type=model_type,
                        model_size=model_size,
                        input_dim=input_dim
                    )
                    
                # Calculate training time
                training_time = time.time() - model_start_time
                
                # Store model if requested
                if save_models:
                    self.trained_models[model_name] = model
                
                # Store results
                self.results[model_name] = {
                    'type': model_type,
                    'size': model_size,
                    'word_error_rate': test_metrics['word_error_rate'],
                    'training_time': training_time,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'num_parameters': sum(p.numel() for p in model.parameters()),
                    'gpu_mem_usage': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
                }
                
                # Reset GPU memory tracking
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                self.results[model_name] = {
                    'type': model_type,
                    'size': model_size,
                    'error': str(e)
                }
        
        # Save results to disk
        self._save_results()
        
        # Return results as DataFrame
        return self.get_results_df()
    
    def get_results_df(self):
        """Convert results dictionary to pandas DataFrame for easier analysis"""
        return pd.DataFrame.from_dict(self.results, orient='index')
    
    def _save_results(self):
        """Save results to JSON file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"comparison_results_{timestamp}.json")
        
        # Convert results to serializable format
        serializable_results = {}
        for model_name, metrics in self.results.items():
            serializable_results[model_name] = {k: float(v) if isinstance(v, (np.float32, torch.Tensor)) else v 
                                                for k, v in metrics.items() if v is not None}
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Also save as CSV for easier analysis
        df = self.get_results_df()
        df.to_csv(os.path.join(self.results_dir, f"comparison_results_{timestamp}.csv"))
            
    def visualize_results(self, 
                          metrics=None, 
                          figsize=(12, 10), 
                          save_path=None):
        """
        Visualize comparison results
        
        Args:
            metrics: List of metrics to visualize (defaults to WER and training time)
            figsize: Figure size
            save_path: Path to save visualization
        """
        if metrics is None:
            metrics = ['word_error_rate', 'training_time']
            
        df = self.get_results_df()
        
        # Check if we have data to visualize
        if df.empty:
            print("No results to visualize")
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]  # Make sure axes is always a list
            
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                # Sort by metric value
                plot_df = df.sort_values(metric)
                
                # Create barplot
                sns.barplot(x=plot_df.index, y=metric, hue='type', data=plot_df, ax=axes[i])
                
                # Customize plot
                axes[i].set_title(f"Comparison of {metric} across LLM Models")
                axes[i].set_xlabel("Model")
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, v in enumerate(plot_df[metric]):
                    axes[i].text(j, v, f"{v:.4f}" if v < 1 else f"{v:.1f}", 
                             ha='center', va='bottom')
            else:
                axes[i].text(0.5, 0.5, f"Metric '{metric}' not available", 
                         ha='center', va='center', transform=axes[i].transAxes)
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
    def compare_generation_examples(self, num_examples=5, models=None):
        """
        Generate and compare outputs from different models on the same inputs
        
        Args:
            num_examples: Number of examples to compare
            models: List of model names to compare (if None, use all trained models)
        """
        if not self.trained_models:
            print("No trained models available")
            return
            
        if models is None:
            models = list(self.trained_models.keys())
        else:
            # Filter to only include models we have
            models = [m for m in models if m in self.trained_models]
            
        if not models:
            print("No specified models available")
            return
            
        # Get a few random samples from the dataset
        sample_indices = np.random.choice(len(self.dataset), num_examples, replace=False)
        
        # Set up the table for comparison
        comparisons = []
        
        device = next(iter(self.trained_models.values())).parameters().__next__().device
        
        for idx in sample_indices:
            input_emb, target_text = self.dataset[idx]
            input_emb = input_emb.unsqueeze(0).to(device)  # Add batch dimension
            
            example_results = {"Target": target_text}
            
            for model_name in models:
                model = self.trained_models[model_name]
                model.eval()
                
                with torch.no_grad():
                    output_text = model(input_emb)[0]  # Get first item from batch
                    example_results[model_name] = output_text
                    
            comparisons.append(example_results)
            
        # Convert to DataFrame for display
        comparisons_df = pd.DataFrame(comparisons)
        
        # Calculate WER for each model
        for model_name in models:
            wer_values = []
            for i in range(len(comparisons)):
                from jiwer import wer
                wer_value = wer(comparisons[i]['Target'], comparisons[i][model_name])
                wer_values.append(wer_value)
            comparisons_df[f"{model_name}_WER"] = wer_values
            
        return comparisons_df
    
    @staticmethod
    def load_saved_results(results_file):
        """Load saved comparison results"""
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        comparator = LLMComparator(None)
        comparator.results = defaultdict(dict, results)
        
        return comparator


def run_basic_comparison(dataset, input_dim=512):
    """
    Run a basic comparison of different LLM architectures
    
    Args:
        dataset: SpeechBCIDataSet_Embedded instance to use for experiments
        input_dim: Dimension of the VQ-VAE codebook vectors
    """
    # Define models to compare
    model_configs = [
        {'name': 't5_small', 'type': 't5', 'size': 'small'},
        {'name': 't5_base', 'type': 't5', 'size': 'base'},
        {'name': 'bart_base', 'type': 'bart', 'size': 'base'},
        {'name': 'gpt2_small', 'type': 'gpt2', 'size': 'small'}
    ]
    
    # Initialize comparator
    comparator = LLMComparator(dataset)
    
    # Run comparison
    results_df = comparator.run_comparison(
        model_configs=model_configs,
        input_dim=input_dim,
        epochs=3,  # Use fewer epochs for a quick comparison
        batch_size=16
    )
    
    # Visualize results
    comparator.visualize_results(
        metrics=['word_error_rate', 'training_time', 'num_parameters'],
        save_path='llm_comparison_results.png'
    )
    
    # Show generation examples
    examples_df = comparator.compare_generation_examples(num_examples=3)
    print("\nGeneration Examples:")
    print(examples_df)
    
    return comparator


if __name__ == "__main__":
    # This is a placeholder for how to use the comparator
    # You'll need to replace this with your actual dataset
    print("To use the LLM comparator, run:")
    print("from llm_comparison import run_basic_comparison")
    print("from your_dataset_module import SpeechBCIDataSet_Embedded")
    print("dataset = SpeechBCIDataSet_Embedded(...)")
    print("comparator = run_basic_comparison(dataset, input_dim=512)")
    
    # Example of how you would extend this with hyperparameter search
    print("\nYou can also perform hyperparameter comparisons:")
    print("comparator = LLMComparator(dataset)")
    print("learning_rates = [1e-5, 3e-5, 5e-5]")
    print("for lr in learning_rates:")
    print("    model_configs = [{'name': f't5_small_lr_{lr}', 'type': 't5', 'size': 'small'}]")
    print("    comparator.run_comparison(model_configs=model_configs, lr=lr)")
    print("comparator.visualize_results()")
