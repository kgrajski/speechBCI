import torch
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm

class ModelDiagnostics:
    """Comprehensive diagnostics for the multimodal LLM"""
    
    def __init__(self, model, adapter, tokenizer, tensorboard_dir, output_dir):
        """
        Initialize the diagnostics module
        
        Args:
            model: The base MMLLM model 
            adapter: The input adapter component
            tokenizer: Tokenizer for decoding outputs
            tensorboard_dir: Directory for tensorboard logs
            output_dir: Directory to save diagnostic outputs
        """
        self.model = model
        self.adapter = adapter  # Store adapter separately
        self.tokenizer = tokenizer
        self.tensorboard_dir = tensorboard_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, "diagnostics"))
        self.attention_patterns = []
        self.output_diversity = []
        
    def capture_epoch_diagnostics(self, epoch, dataloader):
        """Capture diagnostics for a specific epoch"""
        print(f"\nRunning diagnostics for epoch {epoch}...")
        
        # Save current model state
        self.model.eval()
        self.adapter.eval()  # Set adapter to eval mode too
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            # Get a batch
            batch = next(iter(dataloader))
            inputs = batch["vqvae_embeddings"].to(device)
            
            # Process through adapter first - this matches our processing flow
            adapter_outputs = self.adapter(inputs)
            
            # Hook function to capture attention weights
            attention_weights = []
            
            def attn_hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attention_weights.append(output[1].detach().cpu())
            
            # Register hooks on attention layers
            hooks = []
            for name, module in self.model.named_modules():
                if "self_attn" in name and "output" not in name:
                    hooks.append(module.register_forward_hook(attn_hook))
            
            # Forward pass through model using adapter outputs
            outputs = self.model(inputs_embeds=adapter_outputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Analyze adapter outputs directly
            self._analyze_adapter_outputs(adapter_outputs, epoch)
            
            # Store attention patterns for later visualization
            if attention_weights:
                self.attention_patterns.append({
                    'epoch': epoch,
                    'weights': attention_weights
                })
                
            # Generate some text outputs and analyze diversity
            generated_ids = self.model.generate(
                inputs_embeds=adapter_outputs,  # Use adapter outputs
                max_length=30,
                num_beams=5
            )
            
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            self._analyze_text_diversity(decoded, epoch)
    
    def _analyze_adapter_outputs(self, adapter_outputs, epoch):
        """Analyze the adapter outputs for diversity patterns"""
        # Basic statistics
        mean_val = adapter_outputs.mean().item()
        std_val = adapter_outputs.std().item()
        
        # Check for mode collapse
        batch_size = adapter_outputs.shape[0]
        if batch_size > 1:
            # Flatten each example
            flat_outputs = adapter_outputs.view(batch_size, -1)
            
            # Compute pairwise similarity
            norm_outputs = F.normalize(flat_outputs, p=2, dim=1)
            similarity = torch.mm(norm_outputs, norm_outputs.t())
            
            # Remove self-similarity
            mask = torch.ones_like(similarity) - torch.eye(similarity.size(0), device=similarity.device)
            mean_sim = (similarity * mask).sum() / (batch_size * (batch_size - 1))
            
            # Store and log metrics
            self.writer.add_scalar("Adapter/output_mean", mean_val, epoch)
            self.writer.add_scalar("Adapter/output_std", std_val, epoch)
            self.writer.add_scalar("Adapter/cross_similarity", mean_sim.item(), epoch)
            
            # Store for final analysis
            self.output_diversity.append({
                'epoch': epoch,
                'mean_sim': mean_sim.item(),
                'std': std_val
            })
    
    def _analyze_text_diversity(self, decoded_texts, epoch):
        """Analyze diversity in generated text outputs"""
        # Count unique texts and prefixes
        unique_texts = len(set(decoded_texts))
        
        # Get unique starting words
        first_words = []
        for text in decoded_texts:
            words = text.strip().split()
            if words:
                first_words.append(words[0])
        
        unique_first_words = len(set(first_words))
        
        # Log metrics
        self.writer.add_scalar("Text/unique_ratio", unique_texts/len(decoded_texts), epoch)
        self.writer.add_scalar("Text/unique_first_words", unique_first_words, epoch)
    
    def visualize_attention_patterns(self):
        """Visualize attention patterns collected during training using Plotly"""
        if not self.attention_patterns:
            print("No attention patterns collected")
            return
            
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "attention_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot attention patterns for each epoch
        for pattern in self.attention_patterns:
            epoch = pattern['epoch']
            weights = pattern['weights']
            
            # Plot first few attention heads from different layers
            for layer_idx, layer_weights in enumerate(weights[:3]):  # First 3 layers
                # Create figure for multiple attention heads
                n_heads = min(4, layer_weights.shape[1])  # Up to 4 heads
                fig = make_subplots(rows=2, cols=2, 
                                   subplot_titles=[f"Head {i}" for i in range(n_heads)])
                
                # Add attention maps for each head
                for head_idx in range(n_heads):
                    # Get attention map for this head
                    attn_map = layer_weights[0, head_idx].numpy()
                    
                    # Create heatmap
                    row, col = head_idx // 2 + 1, head_idx % 2 + 1
                    fig.add_trace(
                        go.Heatmap(z=attn_map, colorscale='Viridis'),
                        row=row, col=col
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"Layer {layer_idx} Attention Patterns - Epoch {epoch}",
                    height=800,
                    width=1000
                )
                
                # Save as HTML
                html_path = os.path.join(plots_dir, f"attention_epoch{epoch}_layer{layer_idx}.html")
                fig.write_html(html_path)
                
                # Also save a static image for tensorboard
                image_path = os.path.join(plots_dir, f"attention_epoch{epoch}_layer{layer_idx}.png")
                fig.write_image(image_path)
                
                # Add image to tensorboard
                with open(image_path, 'rb') as f:
                    img_bytes = f.read()
                    self.writer.add_image(f"Attention/layer{layer_idx}_epoch{epoch}", 
                                         np.array(img_bytes), dataformats='raw')
    
    def analyze_mode_collapse(self):
        """Analyze if mode collapse occurred during training using Plotly"""
        if not self.output_diversity:
            print("No output diversity data collected")
            return
            
        # Get data from stored diversity metrics
        epochs = [d['epoch'] for d in self.output_diversity]
        similarities = [d['mean_sim'] for d in self.output_diversity]
        stds = [d['std'] for d in self.output_diversity]
        
        # Create figure with two subplots
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=["Cross-sample Similarity", "Output Standard Deviation"])
        
        # Add similarity plot
        fig.add_trace(
            go.Scatter(x=epochs, y=similarities, mode='lines+markers', 
                      name="Cross-sample Similarity"),
            row=1, col=1
        )
        
        # Add threshold line
        fig.add_trace(
            go.Scatter(x=[min(epochs), max(epochs)], y=[0.9, 0.9], 
                      mode='lines', name="Collapse Threshold",
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Add standard deviation plot
        fig.add_trace(
            go.Scatter(x=epochs, y=stds, mode='lines+markers', 
                      name="Output Standard Deviation"),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Mode Collapse Analysis Over Training",
            height=500,
            width=1200
        )
        
        # Set axis titles
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Mean Similarity", row=1, col=1)
        fig.update_yaxes(title_text="Standard Deviation", row=1, col=2)
        
        # Save as HTML
        html_path = os.path.join(self.output_dir, "mode_collapse_analysis.html")
        fig.write_html(html_path)
        
        # Log final analysis
        mode_collapse_risk = "HIGH" if similarities[-1] > 0.8 else "LOW"
        print(f"\nMode Collapse Risk: {mode_collapse_risk}")
        print(f"Final cross-sample similarity: {similarities[-1]:.4f}")
        print(f"Final output standard deviation: {stds[-1]:.4f}")
        
        # Save detailed analysis to text file
        with open(os.path.join(self.output_dir, "mode_collapse_summary.txt"), "w") as f:
            f.write(f"Mode Collapse Risk: {mode_collapse_risk}\n")
            f.write(f"Final cross-sample similarity: {similarities[-1]:.4f}\n")
            f.write(f"Final output standard deviation: {stds[-1]:.4f}\n\n")
            
            if similarities[-1] > 0.8:
                f.write("RECOMMENDATION: Increase diversity_loss_weight to combat mode collapse\n")
                f.write("RECOMMENDATION: Try lower learning rate or different attention mechanisms\n")
            else:
                f.write("Model appears to be maintaining representation diversity\n")
    
    def run_all_diagnostics(self, dataloader):
        """Run comprehensive diagnostics on model and adapter"""
        self.model.eval()
        self.adapter.eval()
        device = next(self.model.parameters()).device
        
        # Analyze outputs from adapter
        print("\nAnalyzing adapter outputs...")
        with torch.no_grad():
            batch = next(iter(dataloader))
            inputs = batch["vqvae_embeddings"].to(device)
            adapter_outputs = self.adapter(inputs)
            
            # Log adapter output statistics
            fig = self._create_adapter_output_visualization(adapter_outputs)
            fig.write_html(os.path.join(self.output_dir, "adapter_output_stats.html"))
        
        # Analyze adapter parameters
        if hasattr(self.adapter, "get_diagnostic_data"):
            print("\nCollecting adapter diagnostic data...")
            _, diag_data = self.adapter.get_diagnostic_data(inputs)
            self._visualize_adapter_diagnostics(diag_data)
        
        # Run standard model diagnostics
        print("\nRunning model output analysis...")
        
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(dataloader, total=min(5, len(dataloader))):
                inputs = batch["vqvae_embeddings"].to(device)
                adapter_outputs = self.adapter(inputs)
                
                generated_ids = self.model.generate(
                    inputs_embeds=adapter_outputs,
                    max_length=30,
                    num_beams=5,
                    do_sample=True
                )
                all_preds.extend(generated_ids.cpu())
                if len(all_preds) >= 50:
                    break
        
        # Continue with output analysis...
    
    def _create_adapter_output_visualization(self, adapter_outputs):
        """Create visualization of adapter output statistics"""
        # Flatten outputs for analysis
        flat_outputs = adapter_outputs.reshape(-1, adapter_outputs.shape[-1])
        
        # Calculate statistics
        mean_values = adapter_outputs.mean(dim=[0, 1]).cpu().numpy()
        std_values = adapter_outputs.std(dim=[0, 1]).cpu().numpy()
        
        # Create distribution plot
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=["Adapter Output Mean per Dimension", 
                                          "Adapter Output Standard Deviation per Dimension"])
        
        # Mean distribution
        fig.add_trace(
            go.Bar(x=list(range(len(mean_values))), 
                   y=mean_values,
                   name="Dimension Mean"),
            row=1, col=1
        )
        
        # Standard deviation distribution
        fig.add_trace(
            go.Bar(x=list(range(len(std_values))), 
                   y=std_values,
                   name="Dimension StdDev"),
            row=2, col=1
        )
        
        fig.update_layout(height=800, width=1000, title_text="Adapter Output Statistics")
        return fig
    
    def _visualize_adapter_diagnostics(self, diag_data):
        """Visualize adapter's internal behavior based on diagnostic data"""
        # Create directory for adapter visualizations
        adapter_viz_dir = os.path.join(self.output_dir, "adapter_internals")
        os.makedirs(adapter_viz_dir, exist_ok=True)
        
        # Visualize layer outputs
        if 'layer_outputs' in diag_data and diag_data['layer_outputs']:
            for i, layer_output in enumerate(diag_data['layer_outputs']):
                # Get first sample from batch for visualization
                sample_output = layer_output[0].cpu().numpy()
                
                # Create heatmap of activations
                fig = px.imshow(
                    sample_output,
                    title=f"Layer {i+1} Output Activation Pattern",
                    labels=dict(x="Sequence Position", y="Feature Dimension"),
                    color_continuous_scale="Viridis"
                )
                fig.write_html(os.path.join(adapter_viz_dir, f"layer_{i+1}_activations.html"))
        
        # Visualize attention weights if available
        if 'attention_weights' in diag_data and diag_data['attention_weights']:
            for i, attn_weights in enumerate(diag_data['attention_weights']):
                # For first sample, first head
                sample_attn = attn_weights[0, 0].cpu().numpy()
                
                # Create attention heatmap
                fig = px.imshow(
                    sample_attn,
                    title=f"Attention Layer {i+1}, Head 0",
                    labels=dict(x="Key Position", y="Query Position"),
                    color_continuous_scale="Viridis"
                )
                fig.write_html(os.path.join(adapter_viz_dir, f"attention_{i+1}_head_0.html"))