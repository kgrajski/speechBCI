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
    """
    Comprehensive diagnostics for the multimodal LLM architecture
    
    This class provides tools to diagnose common issues in multimodal models:
    - Mode collapse/compression (when adapter outputs become too similar)
    - Attention pattern analysis (what the model is attending to)
    - Output diversity assessment (how varied the model's outputs are)
    - Parameter efficiency analysis
    
    These diagnostics help detect problems early in training and suggest
    appropriate remedies before they become entrenched in the model.
    """
    
    def __init__(self, model, adapter, tokenizer, writer, output_dir):
        """
        Initialize the diagnostics module
        
        Args:
            model: The base MMLLM model 
            adapter: The input adapter component
            tokenizer: Tokenizer for decoding outputs
            writer: TensorBoard SummaryWriter instance for logging metrics
            output_dir: Directory to save diagnostic outputs
        """
        self.model = model
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.writer = writer  # Use provided writer instead of creating one
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.attention_patterns = []
        self.output_diversity = []
    
    def _analyze_adapter_outputs(self, adapter_outputs, epoch):
        """
        Analyze the adapter outputs for diversity and mode collapse patterns
        
        This method calculates cross-sample similarity and variance metrics.
        
        Expected healthy values:
        - Mean similarity between samples: <0.6 (lower is better)
        - Output standard deviation: >0.1 (higher is better)
        
        Warning signs:
        - Increasing similarity over epochs (trend toward 1.0)
        - Decreasing standard deviation over epochs (trend toward 0)
        - Mean similarity >0.8 indicates severe mode collapse
        
        Args:
            adapter_outputs: Outputs from the adapter
            epoch: Current epoch number
        """
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
        """
        Analyze diversity in generated text outputs
        
        This tracks unique text ratio and unique first word count.
        
        Expected healthy values:
        - Unique text ratio: >0.8 (higher is better)
        - Unique first words: >5 for small batches, more for larger batches
        
        Warning signs:
        - Unique text ratio decreasing over epochs
        - Very low unique first words count (indicates repetitive starts)
        - All texts starting with the same few words
        
        Args:
            decoded_texts: List of decoded text strings
            epoch: Current epoch number
        """
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
        """
        Visualize attention patterns collected during training using Plotly
        
        Generates interactive heatmaps of attention weights for each layer and head.
        
        What to look for:
        - Healthy patterns: Structured, varied attention across the sequence
        - Diagonal patterns: Model focusing on current token and nearby context
        - Warning signs: 
          - Uniform attention (all values close to 1/sequence_length)
          - Extreme focus on only one or two positions
          - No change in attention patterns across epochs
        """
        if not self.attention_patterns:
            print("No attention patterns collected. Skip visualization.")
            return
            
        os.makedirs(os.path.join(self.output_dir, "attention_plots"), exist_ok=True)
        
        # Process each epoch's attention data
        for att_data in self.attention_patterns:
            epoch = att_data['epoch']
            weights = att_data['weights']
            
            # Process each layer
            for layer_idx, layer_weights in enumerate(weights):
                # Convert attention weights to numpy for visualization
                if isinstance(layer_weights, torch.Tensor):
                    layer_weights = layer_weights.cpu().numpy()
                    
                # Get shape information
                if len(layer_weights.shape) == 4:
                    batch_size, num_heads, seq_len, _ = layer_weights.shape
                else:
                    print(f"Skipping layer with unexpected shape: {layer_weights.shape}")
                    continue
                
                # Create attention heatmap for first example in batch, first head
                fig = go.Figure(data=go.Heatmap(
                    z=layer_weights[0, 0],
                    colorscale='Viridis'
                ))
                
                fig.update_layout(
                    title=f"Attention Pattern - Layer {layer_idx}, Epoch {epoch}",
                    xaxis_title="Key Position",
                    yaxis_title="Query Position"
                )
                
                # Save as HTML file
                output_file = os.path.join(
                    self.output_dir, 
                    "attention_plots", 
                    f"attention_epoch{epoch}_layer{layer_idx}.html"
                )
                fig.write_html(output_file)
                
                # Use add_text instead of add_image - much cleaner approach
                self.writer.add_text(
                    f"Attention/layer{layer_idx}_epoch{epoch}",
                    f"Attention visualization saved to {output_file}",
                    global_step=epoch
                )
    
    def analyze_mode_collapse(self):
        """
        Analyze if mode collapse occurred during training using Plotly
        
        Mode collapse is when the adapter maps different inputs to very similar 
        representations, losing the ability to distinguish input variations.
        
        Plots cross-sample similarity and standard deviation over training.
        
        Interpretation:
        - Healthy training: Similarity stays below 0.8, std.dev remains high
        - Mode collapse: Similarity approaches 1.0, std.dev approaches 0
        - Recovery: Similarity decreases after increasing, std.dev increases
        
        Corrective actions (if collapse detected):
        - Increase diversity_loss_weight in training
        - Reduce adapter complexity
        - Lower the learning rate
        - Add regularization to the adapter
        """
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
    
    def capture_epoch_diagnostics(self, model, dataloader, epoch):
        """
        Capture diagnostic snapshots for a specific training epoch
        
        Args:
            model: The model to diagnose
            dataloader: Data loader to use for diagnostics
            epoch: Current epoch number
        """
        print(f"\nRunning diagnostics for epoch {epoch}...")
        
        model.eval()
        self.adapter.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            # Get a batch
            batch = next(iter(dataloader))
            inputs = batch["vqvae_embeddings"].to(device)
            attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
            labels = batch["label_embeddings"].to(device)
            
            # Process through adapter
            adapter_outputs = self.adapter(inputs)
            
            # IMPORTANT: Use direct method to get attention instead of hooks
            # Forward pass with explicit request for attention weights
            outputs = model(
                inputs_embeds=adapter_outputs,
                attention_mask=attention_mask,
                output_attentions=True,  # Request attention weights
                labels=labels
            )
            
            # Extract attention weights directly from outputs rather than using hooks
            attention_weights = []
            if hasattr(outputs, "encoder_attentions") and outputs.encoder_attentions:
                attention_weights.extend([attn.cpu() for attn in outputs.encoder_attentions])
            if hasattr(outputs, "decoder_attentions") and outputs.decoder_attentions:
                attention_weights.extend([attn.cpu() for attn in outputs.decoder_attentions])
            if hasattr(outputs, "cross_attentions") and outputs.cross_attentions:
                attention_weights.extend([attn.cpu() for attn in outputs.cross_attentions])
            
            # Analyze adapter outputs
            self._analyze_adapter_outputs(adapter_outputs, epoch)
            
            # Store attention patterns for later visualization
            if attention_weights:
                self.attention_patterns.append({
                    'epoch': epoch,
                    'weights': attention_weights
                })
                
            # Generate text outputs with attention mask
            try:
                generated_ids = model.generate(
                    inputs_embeds=adapter_outputs,
                    attention_mask=attention_mask,
                    max_length=30,
                    num_beams=5
                )
                
                decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                self._analyze_text_diversity(decoded, epoch)
            except Exception as e:
                print(f"Warning: Could not generate text during diagnostics: {e}")
    
    def run_all_diagnostics(self, dataloader, model=None):
        """
        Run comprehensive diagnostics on model and adapter
        
        This performs a full battery of tests to evaluate the health of the model-adapter system:
        1. Adapter output analysis (distribution, diversity)
        2. Internal representation visualization (activation patterns)
        3. Generation diversity assessment
        4. Parameter efficiency checks
        
        The visualizations and metrics help identify:
        - Mode collapse issues
        - Attention bottlenecks
        - Representation quality problems
        - Training inefficiencies
        
        Args:
            dataloader: Data loader to use for diagnostics
            model: Optional model to analyze (uses self.model if None)
        """
        # Use provided model or fall back to self.model
        model = model or self.model
        
        model.eval()
        self.adapter.eval()
        device = next(model.parameters()).device
        
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
                
                generated_ids = model.generate(
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
        """
        Create visualization of adapter output statistics
        
        Generates bar charts showing mean and standard deviation for each output dimension.
        
        Healthy indicators:
        - Variable means across dimensions
        - Substantial standard deviation in most dimensions
        - No dominating dimensions (where one dimension has much larger values)
        
        Warning signs:
        - Most dimensions near zero mean with tiny standard deviation
        - A few dimensions with extremely high values (dimension collapse)
        - Very uniform mean values across all dimensions
        
        Args:
            adapter_outputs: Tensor of adapter outputs
            
        Returns:
            Plotly figure with dimension statistics
        """
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
        """
        Visualize adapter's internal behavior based on diagnostic data
        
        Creates heatmaps of layer activations and attention maps.
        
        Healthy patterns:
        - Diverse activation patterns across sequence positions
        - Structured attention patterns that vary by position
        - Progressive abstraction in deeper layers
        
        Warning signs:
        - Highly similar activation patterns for different inputs
        - Uniform attention weights (1/seq_len everywhere)
        - Dead units (dimensions with zero activation)
        - Saturated units (dimensions at max value)
        
        Args:
            diag_data: Dictionary of diagnostic data from adapter
        """
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