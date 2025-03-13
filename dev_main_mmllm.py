import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import evaluate
from jiwer import wer

class VQVAEProjector(nn.Module):
    """
    Projects VQ-VAE embeddings (256-dim) to the LLM embedding space.
    """
    def __init__(self, vqvae_dim=256, llm_dim=768, hidden_dim=512, dropout=0.1):
        super(VQVAEProjector, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(vqvae_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_dim)
        )
        self.pos_embedding = nn.Embedding(16000, vqvae_dim)  # Positional embedding
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, vqvae_dim]
        """
        # Add positional information
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embed = self.pos_embedding(positions)
        
        # Add positional embedding to input
        x = x + pos_embed
        
        # Project to LLM dimension
        return self.projection(x)

class SpeechBCITranslator(nn.Module):
    """
    Complete model for translating VQ-VAE embeddings to English sentences.
    """
    def __init__(self, vqvae_dim=256, model_name="t5-base", max_length=512):
        super(SpeechBCITranslator, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.projector = VQVAEProjector(
            vqvae_dim=vqvae_dim, 
            llm_dim=self.model.config.d_model,
            hidden_dim=512
        )
        self.max_length = max_length
        
    def forward(self, vqvae_embeddings, attention_mask=None, labels=None):
        # Project VQ-VAE embeddings to LLM embedding space
        projected_embeddings = self.projector(vqvae_embeddings)
        
        if labels is not None:
            # Training mode
            outputs = self.model(
                inputs_embeds=projected_embeddings,
                attention_mask=attention_mask,
                labels=labels
            )
            return outputs
        else:
            # Inference mode
            outputs = self.model.generate(
                inputs_embeds=projected_embeddings,
                attention_mask=attention_mask,
                max_length=100,  # Max 100 words as specified
                num_beams=4,
                early_stopping=True
            )
            return outputs
    
    def save(self, path):
        """Save the model to disk."""
        torch.save({
            'projector': self.projector.state_dict(),
            'model': self.model.state_dict()
        }, path)
    
    def load(self, path):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.projector.load_state_dict(checkpoint['projector'])
        self.model.load_state_dict(checkpoint['model'])

def process_long_sequence(model, sequence, max_length=512, strategy="chunk"):
    """
    Processes sequences longer than the model's context window.
    
    Args:
        model: The model to use for processing
        sequence (torch.Tensor): Input sequence
        max_length (int): Maximum sequence length
        strategy (str): Strategy to use ('chunk', 'compress', or 'hierarchical')
    """
    if strategy == "compress":
        # Compression strategy: average adjacent vectors
        compression_factor = max(1, len(sequence) // max_length + 1)
        compressed = []
        for i in range(0, len(sequence), compression_factor):
            chunk = sequence[i:i + compression_factor]
            if len(chunk) > 0:
                compressed.append(torch.mean(chunk, dim=0))
        compressed_seq = torch.stack(compressed)
        if len(compressed_seq) <= max_length:
            return model(compressed_seq.unsqueeze(0))
        else:
            return process_long_sequence(model, compressed_seq, max_length, "chunk")
    
    elif strategy == "chunk":
        # Chunking strategy: process overlapping segments
        chunk_size = max_length
        overlap = min(100, max_length // 4)
        chunks = []
        
        for i in range(0, len(sequence), chunk_size - overlap):
            end = min(i + chunk_size, len(sequence))
            chunks.append(sequence[i:end])
        
        outputs = []
        for chunk in chunks:
            output = model(chunk.unsqueeze(0))
            outputs.append(model.tokenizer.decode(output[0], skip_special_tokens=True))
        
        return " ".join(outputs)
    
    elif strategy == "hierarchical":
        # First compress, then chunk if needed
        return process_long_sequence(model, sequence, max_length, "compress")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def run_training(model, train_dl, val_dl, optimizer, device, num_epochs=3, 
                model_dir=None, tensorboard_dir=None):
    """
    Trains the model on the training set and validates on the validation set.
    """
    print("Starting training...")
    
    # Set up learning rate scheduler
    total_steps = len(train_dl) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    best_val_wer = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_dl), total=len(train_dl))
        
        for step, batch in progress_bar:
            # Get batch data
            vqvae_embeddings, sentences = batch
            vqvae_embeddings = vqvae_embeddings.to(device)
            
            # Create attention mask (all ones for now)
            attention_mask = torch.ones(
                vqvae_embeddings.shape[0], 
                vqvae_embeddings.shape[1],
                device=device
            )
            
            # Tokenize target sentences
            tokenized_output = model.tokenizer(
                sentences, 
                padding=True, 
                truncation=True, 
                max_length=100,
                return_tensors="pt"
            )
            labels = tokenized_output.input_ids.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                vqvae_embeddings, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_description(
                f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}"
            )
        
        avg_train_loss = epoch_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_train_loss:.4f}")
        
        # Evaluate on validation set
        if val_dl is not None:
            val_wer, _ = run_evaluation(model, val_dl, device)
            print(f"Validation WER: {val_wer:.4f}")
            
            # Save best model
            if val_wer < best_val_wer:
                best_val_wer = val_wer
                if model_dir is not None:
                    model.save(os.path.join(model_dir, "best_model.pt"))
        
        # Save checkpoint
        if model_dir is not None:
            model.save(os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    return model

def run_evaluation(model, test_dl, device, verbose=True):
    """
    Evaluates the model on the test set.
    """
    model.eval()
    all_predictions = []
    all_references = []
    all_wers = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating"):
            # Get batch data
            vqvae_embeddings, references = batch
            vqvae_embeddings = vqvae_embeddings.to(device)
            
            # Handle long sequences
            batch_predictions = []
            for i in range(len(vqvae_embeddings)):
                if vqvae_embeddings[i].shape[0] > model.max_length:
                    # Process long sequence
                    prediction = process_long_sequence(
                        model, 
                        vqvae_embeddings[i], 
                        max_length=model.max_length
                    )
                    batch_predictions.append(prediction)
                else:
                    # Standard processing
                    attention_mask = torch.ones(
                        1, vqvae_embeddings[i].shape[0], device=device
                    )
                    output_ids = model(
                        vqvae_embeddings[i].unsqueeze(0), 
                        attention_mask=attention_mask
                    )
                    prediction = model.tokenizer.decode(
                        output_ids[0], skip_special_tokens=True
                    )
                    batch_predictions.append(prediction)
            
            # Calculate WER for each example
            for pred, ref in zip(batch_predictions, references):
                wer_score = wer(ref, pred)
                all_wers.append(wer_score)
            
            all_predictions.extend(batch_predictions)
            all_references.extend(references)
    
    # Calculate overall WER
    overall_wer = wer(all_references, all_predictions)
    
    # Detailed results
    results = {
        "overall_wer": overall_wer,
        "per_example_wer": all_wers,
        "mean_wer": np.mean(all_wers),
        "median_wer": np.median(all_wers),
        "std_wer": np.std(all_wers),
        "min_wer": np.min(all_wers),
        "max_wer": np.max(all_wers)
    }
    
    if verbose:
        print(f"Overall WER: {overall_wer:.4f}")
        print(f"Mean WER: {results['mean_wer']:.4f}")
        print(f"Median WER: {results['median_wer']:.4f}")
        print(f"Min WER: {results['min_wer']:.4f}")
        print(f"Max WER: {results['max_wer']:.4f}")
        
    return overall_wer, results