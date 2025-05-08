import os
import numpy as np
import torch
import umap
from plotly.subplots import make_subplots
from tqdm import tqdm
import plotly.graph_objects as go


class MMLLM_Diagnostics:
    
    @staticmethod
    def _apply_padding_mask(input_tensor, padding_mask):
            """
            Apply padding mask to input tensor by filtering out padded positions.
            
            Args:
                input_tensor (torch.Tensor): Input tensor of shape (T, D) where T is sequence length
                    and D is feature dimension
                padding_mask (torch.Tensor): Padding mask of shape (T,) where 1 indicates valid
                    positions and 0 indicates padding
                    
            Returns:
                torch.Tensor: Tensor containing only non-padded positions, shape (T_trimmed, D)
                    where T_trimmed is the number of valid (non-padded) positions
            """
            # Get indices of non-padded positions (where mask is 1)
            valid_indices = torch.nonzero(padding_mask, as_tuple=True)[0]
            # Select only the valid positions from input tensor
            return input_tensor[valid_indices]

    #
    # This method is designed to save the embedded inputs and corresponding adapter outputs.
    #
    def embed_encode_comp_plots(self, mmllm, dataloader, epoch, output_dir, device):
        """
        Save embedded inputs and corresponding adapter outputs for test trials.
        This method is designed to track how the same test trials evolve across epochs.
        
        Args:
            model: The model to evaluate
            dataloader: Test dataloader (should be deterministic, same order each epoch)
            epoch: Current training epoch
            output_dir: Directory to save visualizations
            device: Device to run computations on
        """
        mmllm.base_model.eval()  # Make sure model is in eval mode
        mmllm.input_adapter.eval()  # Make sure encoder is in eval mode

        # Create a directory for the embed-encode data
        embed_encode_dir = os.path.join(output_dir, "embed_encode_data")
        os.makedirs(embed_encode_dir, exist_ok=True)

        # Process all test batches - using test_dl ensures same trials each epoch
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing test trials")):
            inputs = batch["vqvae_embeddings"].to(device)
            padding_masks = batch["padding_masks"].to(device)
            positional_encodings = batch["positional_encodings"].to(device)
            labels = batch["label_embeddings"].to(device)
            labels_text = batch["original_text"]
            trial_ids = batch["trial_id"]

            # Forward pass through model - returns encoder outputs and all losses
            adapter_outputs, losses = mmllm(inputs, padding_masks, positional_encodings, labels)

            # For each trial_id pair separately apply UMAP to the inputs and adapter outputs
            # and save the UMAP as side-by-side plots in a single html file
            for i, trial_id in enumerate(trial_ids):
                inputs_masked = self._apply_padding_mask(inputs[i], padding_masks[i])
                adapter_outputs_masked = self._apply_padding_mask(adapter_outputs[i], padding_masks[i])
                fname = self.create_trial_adapter_comparison(
                    trial_id,
                    labels[i],
                    labels_text[i],
                    inputs_masked,
                    adapter_outputs_masked,
                    epoch,
                    embed_encode_dir,)
                print(f"Saved UMAP comparison to {fname}")

    def create_trial_adapter_comparison(
            self, 
            trial_id,
            labels,
            labels_text,
            input_trial_data,
            adapter_outputs,
            epoch,
            output_dir
    ):
        """
        Create a side-by-side comparison of input trial data and adapter outputs using UMAP.
        
        Args:
            input_trial_data (torch.Tensor): Input trial data tensor
            adapter_outputs (torch.Tensor): Encoder output tensor
            trial_id (str): Identifier for the trial
            epoch (int): Current training epoch
            labels (str): Label for the trial (e.g., spoken text)
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Input Trial Data", "Encoder Outputs"],
            horizontal_spacing=0.1,
            vertical_spacing=0.1,  # Add some vertical spacing
        )

        # Process input trial data - fit UMAP specific to this trial
        input_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        input_numpy = input_trial_data.detach().cpu().numpy()

        # Reshape if needed (flatten any extra dimensions)
        if len(input_numpy.shape) > 2:
            T = input_numpy.shape[0]
            input_numpy = input_numpy.reshape(T, -1)

        # Fit UMAP for input data
        input_umap.fit(input_numpy)
        input_embedding = input_umap.transform(input_numpy)
        n_points_input = input_embedding.shape[0]

        # Process adapter outputs - fit UMAP specific to this trial
        adapter_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        adapter_numpy = adapter_outputs.detach().cpu().numpy()

        # Reshape if needed
        if len(adapter_numpy.shape) > 2:
            T = adapter_numpy.shape[0]
            adapter_numpy = adapter_numpy.reshape(T, -1)

        # Fit UMAP for adapter outputs
        adapter_umap.fit(adapter_numpy)
        adapter_embedding = adapter_umap.transform(adapter_numpy)
        n_points_adapter = adapter_embedding.shape[0]

        # Add input data traces
        # Line for trajectory
        fig.add_trace(
            go.Scatter(
                x=input_embedding[:, 0],
                y=input_embedding[:, 1],
                mode="lines",
                line=dict(color="rgba(100,100,100,0.5)", width=1),
                hoverinfo="none",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Scatter points
        fig.add_trace(
            go.Scatter(
                x=input_embedding[:, 0],
                y=input_embedding[:, 1],
                mode="markers",
                marker=dict(
                    color=np.arange(n_points_input),
                    colorscale="Viridis",
                    size=6,
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(title="Time", x=0.46),
                ),
                text=[f"Step: {i}" for i in range(n_points_input)],
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Start point
        fig.add_trace(
            go.Scatter(
                x=[input_embedding[0, 0]],
                y=[input_embedding[0, 1]],
                mode="markers",
                marker=dict(color="green", size=12, line=dict(color="black", width=1)),
                name="Start",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # End point
        fig.add_trace(
            go.Scatter(
                x=[input_embedding[-1, 0]],
                y=[input_embedding[-1, 1]],
                mode="markers",
                marker=dict(color="red", size=12, line=dict(color="black", width=1)),
                name="End",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Add arrows for input data
        for i in range(0, n_points_input - 1, max(1, n_points_input // 4)):
            fig.add_annotation(
                x=input_embedding[i + 1, 0],
                y=input_embedding[i + 1, 1],
                ax=input_embedding[i, 0],
                ay=input_embedding[i, 1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="black",
                row=1,
                col=1,
            )

        # Add adapter output traces
        # Line for trajectory
        fig.add_trace(
            go.Scatter(
                x=adapter_embedding[:, 0],
                y=adapter_embedding[:, 1],
                mode="lines",
                line=dict(color="rgba(100,100,100,0.5)", width=1),
                hoverinfo="none",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Scatter points
        fig.add_trace(
            go.Scatter(
                x=adapter_embedding[:, 0],
                y=adapter_embedding[:, 1],
                mode="markers",
                marker=dict(
                    color=np.arange(n_points_adapter),
                    colorscale="Viridis",
                    size=6,
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(title="Time", x=1.0),
                ),
                text=[f"Step: {i}" for i in range(n_points_adapter)],
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Start point
        fig.add_trace(
            go.Scatter(
                x=[adapter_embedding[0, 0]],
                y=[adapter_embedding[0, 1]],
                mode="markers",
                marker=dict(color="green", size=12, line=dict(color="black", width=1)),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # End point
        fig.add_trace(
            go.Scatter(
                x=[adapter_embedding[-1, 0]],
                y=[adapter_embedding[-1, 1]],
                mode="markers",
                marker=dict(color="red", size=12, line=dict(color="black", width=1)),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Add arrows for adapter outputs
        for i in range(0, n_points_adapter - 1, max(1, n_points_adapter // 10)):
            fig.add_annotation(
                x=adapter_embedding[i + 1, 0],
                y=adapter_embedding[i + 1, 1],
                ax=adapter_embedding[i, 0],
                ay=adapter_embedding[i, 1],
                xref="x2",
                yref="y2",
                axref="x2",
                ayref="y2",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="black",
                row=1,
                col=2,
            )

        # Update layout
        title = f"Trial ID: {trial_id} (Epoch {epoch})"
        title += f"<br>Label: '{labels}'"
        title += f"<br>Label Text: '{labels_text}'"
        
        fig.update_layout(
            title=title,
            showlegend=True,
            width=1200,
            height=600,
            margin=dict(t=150),  # Increased top margin to 150 pixels
            xaxis=dict(title=None, showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(title=None, showticklabels=False, showgrid=False, zeroline=False),
            xaxis2=dict(title=None, showticklabels=False, showgrid=False, zeroline=False),
            yaxis2=dict(title=None, showticklabels=False, showgrid=False, zeroline=False),
            hovermode="closest",
            # Move subplot titles to bottom
            annotations=[
                dict(
                    x=0.225,  # Position for first subplot
                    y=-0.1,   # Below the plot
                    xref='paper',
                    yref='paper',
                    text="Input Trial Data",
                    showarrow=False,
                    font=dict(size=12)
                ),
                dict(
                    x=0.775,  # Position for second subplot
                    y=-0.1,   # Below the plot
                    xref='paper',
                    yref='paper',
                    text="Encoder Outputs",
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
        )

        # Save the plot
        filename = os.path.join(output_dir, f'umap_comparison_trial_{trial_id}_epoch_{epoch}.html')
        fig.write_html(filename)
        
        return filename

    def attention_heatmap(self, mmllm, dataloader, epoch, output_dir, device, layer_idx=0, head_idx=0):
        """
        Visualize average attention weights for a given layer and head.
        """
        output_dir = os.path.join(output_dir, "attention_heatmap")
        os.makedirs(output_dir, exist_ok=True)
        
        mmllm.base_model.eval()
        attention_weights = []
        for batch in dataloader:
            inputs = batch["vqvae_embeddings"].to(device)
            padding_masks = batch["padding_masks"].to(device)
            positional_encodings = batch["positional_encodings"].to(device)
            labels = batch["label_embeddings"].to(device)
            with torch.no_grad():
                # Forward pass with output_attentions=True
                outputs = mmllm.base_model(
                    inputs_embeds=mmllm.input_adapter(inputs, padding_masks),
                    attention_mask=padding_masks,
                    labels=labels,
                    output_attentions=True,
                )
                # Get encoder attentions: list of [batch, heads, seq, seq]
                attn = outputs.encoder_attentions[layer_idx][:, head_idx].cpu().numpy()  # [batch, seq, seq]
                attention_weights.append(attn)
        # Average over batches
        avg_attn = np.mean(np.concatenate(attention_weights, axis=0), axis=0)
        fig = go.Figure(data=go.Heatmap(z=avg_attn))
        fig.update_layout(title=f"Attention Heatmap Layer {layer_idx} Head {head_idx} (Epoch {epoch})")
        filename = os.path.join(output_dir, f"attention_heatmap_layer{layer_idx}_head{head_idx}_epoch{epoch}.html")
        fig.write_html(filename)
        print(f"Saved attention heatmap to {filename}")

    def weight_heatmap(self, mmllm, epoch, output_dir, module_name="input_adapter"):
        """
        Visualize the weights of a given module as a heatmap.
        """
        output_dir = os.path.join(output_dir, "weight_heatmap")
        os.makedirs(output_dir, exist_ok=True)
        module = getattr(mmllm, module_name)
        for name, param in module.named_parameters():
            if param.ndim == 2:  # Only plot 2D weights
                weights = param.detach().cpu().numpy()
                fig = go.Figure(data=go.Heatmap(z=weights))
                fig.update_layout(title=f"{module_name}.{name} Weights (Epoch {epoch})")
                filename = os.path.join(output_dir, f"weight_heatmap_{module_name}_{name}_epoch{epoch}.html")
                fig.write_html(filename)
                print(f"Saved weight heatmap to {filename}")

    def adapter_activation_histogram(self, mmllm, dataloader, epoch, output_dir, device):
        """
        Plot histogram of adapter activations.
        """
        output_dir = os.path.join(output_dir, "adapter_activation_histogram")
        os.makedirs(output_dir, exist_ok=True)
        mmllm.input_adapter.eval()
        activations = []
        for batch in dataloader:
            inputs = batch["vqvae_embeddings"].to(device)
            padding_masks = batch["padding_masks"].to(device)
            with torch.no_grad():
                out = mmllm.input_adapter(inputs, padding_masks)
                activations.append(out.cpu().numpy())
        activations = np.concatenate([a.flatten() for a in activations])
        fig = go.Figure(data=[go.Histogram(x=activations, nbinsx=100)])
        fig.update_layout(title=f"Adapter Activation Histogram (Epoch {epoch})")
        filename = os.path.join(output_dir, f"adapter_activation_histogram_epoch{epoch}.html")
        fig.write_html(filename)
        print(f"Saved adapter activation histogram to {filename}")

    def position_encoding_heatmap(self, dataloader, output_dir):
        """
        Plot a heatmap of the positional encoding matrix from the dataset.
        The x-axis is the position (sequence index), the y-axis is the embedding dimension.
        """
        import plotly.graph_objects as go
        # Create output directory
        output_dir = os.path.join(output_dir, "position_encoding_heatmap")
        os.makedirs(output_dir, exist_ok=True)

        # Get a batch and extract the positional encoding matrix
        batch = next(iter(dataloader))
        # Assume positional_encodings is [seq_len, embed_dim] or [batch, seq_len, embed_dim]
        pos_enc = batch["positional_encodings"]
        if pos_enc.ndim == 3:
            # Take the first sample in the batch
            pos_enc = pos_enc[0]
        pos_enc_np = pos_enc.detach().cpu().numpy()

        fig = go.Figure(data=go.Heatmap(
            z=pos_enc_np.T,  # Transpose so x=position, y=embedding dim
            colorscale="Viridis"
        ))
        fig.update_layout(
            title=f"Positional Encoding Heatmap)",
            xaxis_title="Position (Sequence Index)",
            yaxis_title="Embedding Dimension",
            width=900,
            height=500
        )
        filename = os.path.join(output_dir, f"position_encoding_heatmap_epoch.html")
        fig.write_html(filename)
        print(f"Saved positional encoding heatmap to {filename}")
