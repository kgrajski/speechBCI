import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from jiwer import wer
from tqdm import tqdm

"""
IMPORTANT NOTE: This implementation strongly recommends using the actual VQ-VAE codebook vectors 
rather than one-hot encoded vectors as input. Using codebook vectors provides:
1. Better memory efficiency (much smaller input tensors)
2. Reduced computational cost (smaller projection matrices)
3. Preservation of semantic information from the VQ-VAE embedding space
4. Better generalization capabilities

The input_dim parameter should be the dimension of each codebook vector (e.g., 512), 
NOT the size of the codebook itself (e.g., 1024).

LLM SELECTION CONSIDERATIONS:

T5 was selected for this implementation for the following reasons:
1. Encoder-decoder architecture: Ideal for sequence-to-sequence tasks where you're mapping from
   one modality (embedded neural signals) to another (English text)
2. Direct embedding input: T5 allows passing custom embeddings via inputs_embeds parameter
3. Scalability: Available in multiple sizes (t5-small: 60M, t5-base: 220M, t5-large: 770M) to
   fit computational constraints
4. Strong text generation capabilities: Pre-trained on diverse text corpora
5. Well-documented and supported in the transformers library

Key assumptions when using T5:
1. The task is primarily sequence-to-sequence mapping (not classification or embedding generation)
2. Data quantity is sufficient for fine-tuning but not for training from scratch
3. Computational resources are significant but not unlimited
4. English language output is the primary target
5. The input sequence length is manageable (typically under 512 tokens)

Alternative LLMs to consider:
1. BART: More focused on sequence-to-sequence tasks and particularly strong at text comprehension
   - Pros: Strong at understanding context, effective for summarization and translation
   - Cons: Generally larger than T5 equivalents, slower inference
   - Use when: Higher fidelity to source content is needed

2. LLaMA: Open-source decoder-only architecture with strong generalization capabilities
   - Pros: State-of-the-art performance, lighter versions available (7B)
   - Cons: Requires special handling for encoder-decoder tasks, higher compute requirements
   - Use when: You need stronger language modeling or have access to substantial compute

3. GPT-2/GPT-3: Decoder-only architectures with powerful text generation
   - Pros: Excellent text generation quality, wide range of sizes
   - Cons: Lacks built-in encoder, requires adaptation for sequence-to-sequence tasks
   - Use when: Output text quality and fluency are paramount

4. mT5/mBART: Multilingual variants of T5/BART
   - Pros: Support for multiple languages, similar architecture to T5/BART
   - Cons: Larger model size due to multilingual capabilities
   - Use when: Non-English output is required

5. RWKV: Alternative architecture combining RNN and transformer characteristics
   - Pros: Linear scaling with sequence length, lower memory footprint
   - Cons: Less mature ecosystem, fewer pretrained models
   - Use when: Very long sequences need to be processed efficiently
"""

class EmbeddingProjector(nn.Module):
    """
    Projects VQ-VAE embeddings to LLM token embeddings with positional encoding.
    """
    def __init__(self, input_dim, model_dim, max_seq_len=512):
        super().__init__()
        self.projection = nn.Linear(input_dim, model_dim)
        
        # Positional encoding
        self.register_buffer(
            "positional_encoding",
            self._create_sinusoidal_encoding(max_seq_len, model_dim)
        )
        self.max_seq_len = max_seq_len
        
    def _create_sinusoidal_encoding(self, max_seq_len, model_dim):
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(np.log(10000.0) / model_dim))
        
        pos_encoding = torch.zeros(max_seq_len, model_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        """
        seq_len = x.size(1)
        # Apply linear projection
        projected = self.projection(x)
        
        # Add positional encoding
        projected = projected + self.positional_encoding[:, :seq_len, :]
        
        return projected

class MultiModalLLM(nn.Module):
    """
    End-to-end model for multimodal sequence to sequence tasks
    
    IMPORTANT: The input_embeddings should be the actual codebook vectors from VQ-VAE
    (not one-hot encoded vectors) for optimal performance and efficiency.
    """
    def __init__(self, input_dim, model_name="t5-small"):
        """
        Args:
            input_dim: Dimension of each VQ-VAE codebook vector (NOT the codebook size)
            model_name: Name of the pretrained T5 model to use
        """
        super().__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.projector = EmbeddingProjector(input_dim, self.t5_model.config.d_model)
        
    def forward(self, input_embeddings, attention_mask=None, target_texts=None):
        """
        input_embeddings: [batch_size, seq_len, input_dim]
        attention_mask: [batch_size, seq_len] binary mask (1 for real, 0 for padding)
        target_texts: list of strings or None
        """
        # Project input embeddings to T5's embedding space with positional encoding
        projected_embeddings = self.projector(input_embeddings)
        
        # Create a default attention mask if none is provided
        if attention_mask is None:
            attention_mask = torch.ones(input_embeddings.shape[0], input_embeddings.shape[1], device=input_embeddings.device)
        
        if target_texts is not None:
            # Training mode
            target_encodings = self.tokenizer(
                target_texts,
                padding="longest",
                return_tensors="pt", 
                truncation=True
            ).to(input_embeddings.device)
            
            # Pass projected embeddings directly as hidden states to encoder
            outputs = self.t5_model(
                inputs_embeds=projected_embeddings,
                attention_mask=attention_mask,
                labels=target_encodings.input_ids,
                decoder_attention_mask=target_encodings.attention_mask
            )
            return outputs.loss
        else:
            # Inference mode
            encoder_outputs = self.t5_model.encoder(
                inputs_embeds=projected_embeddings,
                attention_mask=attention_mask
            )
            
            # Generate text using encoder outputs
            generated_ids = self.t5_model.generate(
                encoder_outputs=encoder_outputs,
                max_length=100,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the generated tokens
            generated_texts = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            return generated_texts

def variable_length_collate_fn(batch):
    """
    Custom collate function for handling variable length sequences of VQ-VAE embeddings
    
    Args:
        batch: List of tuples (input_embedding, target_text)
        
    Returns:
        input_embeddings: Tensor of shape [batch_size, max_seq_len, embedding_dim]
        attention_mask: Binary tensor of shape [batch_size, max_seq_len]
        target_texts: List of target texts
    """
    # Sort batch by sequence length in descending order (optional optimization)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    # Extract inputs and targets
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Get max sequence length in this batch
    max_len = max([inp.shape[0] for inp in inputs])
    
    # Get embedding dimension from the first item
    embed_dim = inputs[0].shape[-1]
    
    # Prepare the output tensors
    batch_size = len(inputs)
    padded_inputs = torch.zeros(batch_size, max_len, embed_dim)
    attention_mask = torch.zeros(batch_size, max_len)
    
    # Fill in the actual data and create mask
    for i, inp in enumerate(inputs):
        seq_len = inp.shape[0]
        padded_inputs[i, :seq_len] = inp
        attention_mask[i, :seq_len] = 1.0  # 1 for real tokens, 0 for padding
    
    return padded_inputs, attention_mask, targets

def train_multimodal_llm(
    dataset,
    input_dim,  # Should be the dimension of the codebook vectors (e.g., 512), not the codebook size
    train_ratio=0.8,
    batch_size=32,
    epochs=5,
    lr=5e-5,
    model_name="t5-small",
    save_path="./multimodal_llm"
):
    """
    Fine-tune a multimodal LLM using SpeechBCIDataSet_Embedded
    
    Args:
        dataset: SpeechBCIDataSet_Embedded instance, should provide VQ-VAE codebook vectors, not one-hot vectors
        input_dim: dimension of input embeddings (the codebook vector dimension, NOT codebook size)
        train_ratio: ratio of data to use for training vs testing
        batch_size: batch size for training
        epochs: number of training epochs
        lr: learning rate
        model_name: name of the base LLM model
        save_path: path to save fine-tuned model
    
    Returns:
        model: trained model
        test_metrics: dictionary with test metrics
    """
    # Split dataset into training and testing sets
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=variable_length_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=variable_length_collate_fn
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Note: When instantiating the model, input_dim should be the VQ-VAE embedding dimension,
    # not the size of the codebook or one-hot vector length
    model = MultiModalLLM(input_dim, model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, attention_mask, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets
            
            optimizer.zero_grad()
            loss = model(inputs, attention_mask, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(batch_idx+1)}")
    
    # Evaluate on test set
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, attention_mask, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            predictions = model(inputs, attention_mask)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    word_error_rate = wer(all_targets, all_predictions)
    
    # Save the model
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)
    
    # Return test metrics
    test_metrics = {
        "word_error_rate": word_error_rate,
        "num_samples": len(all_targets),
    }
    
    return model, test_metrics

def evaluate_on_validation(model, validation_dataset, batch_size=32):
    """
    Evaluate model on validation dataset
    
    Args:
        model: trained MultiModalLLM model
        validation_dataset: SpeechBCIDataSet_Embedded for validation
        batch_size: batch size for evaluation
        
    Returns:
        metrics: dictionary with validation metrics
    """
    device = next(model.parameters()).device
    val_loader = DataLoader(validation_dataset, batch_size=batch_size)
    
    model.eval()
    all_predictions = []
    all_targets = []
    individual_wers = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating"):
            inputs = inputs.to(device)
            predictions = model(inputs)
            
            # Calculate per-sample WER
            for pred, target in zip(predictions, targets):
                sample_wer = wer([target], [pred])
                individual_wers.append(sample_wer)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # Calculate overall WER
    overall_wer = wer(all_targets, all_predictions)
    
    metrics = {
        "word_error_rate": overall_wer,
        "individual_wers": individual_wers,
        "num_samples": len(all_targets),
        "mean_sample_wer": np.mean(individual_wers),
        "median_sample_wer": np.median(individual_wers),
        "std_sample_wer": np.std(individual_wers)
    }
    
    return metrics

# Add a function to help users select alternative LLMs
def get_multimodal_llm(model_type="t5", model_size="small", input_dim=512):
    """
    Factory function to create different types of multimodal LLMs
    
    Args:
        model_type: Type of LLM to use ("t5", "bart", "llama", "gpt2", etc.)
        model_size: Size variant of the model ("small", "base", "large", etc.)
        input_dim: Dimension of the input embeddings from VQ-VAE
        
    Returns:
        A MultiModalLLM instance using the specified base model
    """
    if model_type.lower() == "t5":
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        model_name = f"t5-{model_size}"
        base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    elif model_type.lower() == "bart":
        from transformers import BartForConditionalGeneration, BartTokenizer
        model_name = f"facebook/bart-{model_size}"
        base_model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
    
    elif model_type.lower() == "llama":
        try:
            from transformers import LlamaForCausalLM, LlamaTokenizer
            # Note: Requires special access to Meta's LLaMA models or open-source variants
            model_name = f"meta-llama/Llama-2-{model_size}-hf"  # requires access
            base_model = LlamaForCausalLM.from_pretrained(model_name)
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            print("Warning: LLaMA uses a decoder-only architecture and requires special handling")
        except:
            raise ValueError("LLaMA models require Hugging Face access approval or open-source alternatives")
    
    elif model_type.lower() == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        if model_size == "small":
            model_name = "gpt2"
        else:
            model_name = f"gpt2-{model_size}"
        base_model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("Warning: GPT-2 uses a decoder-only architecture and requires special handling")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create a custom MultiModalLLM with the selected base model
    return CustomMultiModalLLM(base_model, tokenizer, input_dim)

class CustomMultiModalLLM(nn.Module):
    """
    Flexible wrapper for different LLM architectures for multimodal tasks
    This can handle both encoder-decoder models (T5, BART) and decoder-only models (GPT, LLaMA)
    """
    def __init__(self, base_model, tokenizer, input_dim):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.model_type = base_model.__class__.__name__
        
        # Check if model has encoder-decoder structure or decoder-only
        self.has_encoder = hasattr(base_model, "encoder")
        
        # Determine the hidden size based on model's configuration
        if hasattr(base_model, "config"):
            if hasattr(base_model.config, "d_model"):
                hidden_size = base_model.config.d_model
            elif hasattr(base_model.config, "hidden_size"):
                hidden_size = base_model.config.hidden_size
            else:
                # Default fallback
                hidden_size = 768
        
        # Create the projection layer
        self.projector = EmbeddingProjector(input_dim, hidden_size)
    
    def forward(self, input_embeddings, target_texts=None):
        """
        Unified interface for both encoder-decoder and decoder-only models
        """
        # Project input embeddings to the LLM's embedding space with positional encoding
        projected_embeddings = self.projector(input_embeddings)
        
        # Handle differently based on model architecture
        if self.has_encoder:  # Encoder-decoder models (T5, BART)
            # Similar to the original implementation for T5
            # ...existing implementation for encoder-decoder...
            return self._forward_encoder_decoder(projected_embeddings, target_texts)
        else:  # Decoder-only models (GPT-2, LLaMA)
            # Different approach for decoder-only models
            return self._forward_decoder_only(projected_embeddings, target_texts)
    
    def _forward_encoder_decoder(self, projected_embeddings, target_texts=None):
        """Handle encoder-decoder models like T5, BART"""
        if target_texts is not None:
            # Training mode
            target_encodings = self.tokenizer(
                target_texts, 
                padding="longest", 
                return_tensors="pt", 
                truncation=True
            ).to(projected_embeddings.device)
            
            outputs = self.base_model(
                inputs_embeds=projected_embeddings,
                labels=target_encodings.input_ids,
                decoder_attention_mask=target_encodings.attention_mask
            )
            return outputs.loss
        else:
            # Inference mode
            encoder_outputs = self.base_model.encoder(inputs_embeds=projected_embeddings)
            
            generated_ids = self.base_model.generate(
                encoder_outputs=encoder_outputs,
                max_length=100,
                num_beams=4,
                early_stopping=True
            )
            
            generated_texts = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            return generated_texts
    
    def _forward_decoder_only(self, projected_embeddings, target_texts=None):
        """Handle decoder-only models like GPT-2, LLaMA"""
        if target_texts is not None:
            # For decoder-only models, we need to prepend the embedded inputs
            # to the target texts as context, with special handling
            target_encodings = self.tokenizer(
                target_texts,
                padding="longest",
                return_tensors="pt",
                truncation=True
            ).to(projected_embeddings.device)
            
            # Create attention masks that allow seeing projected embeddings
            # but prevent target tokens from attending to future tokens
            seq_len = projected_embeddings.size(1)
            target_len = target_encodings.input_ids.size(1)
            
            # Create a causal mask for the target part
            target_mask = torch.triu(
                torch.ones(target_len, target_len, device=projected_embeddings.device) * -1e10, 
                diagonal=1
            )
            
            # Full attention for projected embeddings part, causal for target part
            full_mask = torch.zeros(
                seq_len + target_len, 
                seq_len + target_len, 
                device=projected_embeddings.device
            )
            full_mask[seq_len:, seq_len:] = target_mask
            
            # Model forward pass with custom handling
            outputs = self.base_model(
                inputs_embeds=projected_embeddings,
                labels=target_encodings.input_ids,
                attention_mask=full_mask
            )
            return outputs.loss
        else:
            # For generation, we'll use the projected embeddings as prefix/prompt
            # and let the model continue generating from there
            outputs = self.base_model(inputs_embeds=projected_embeddings)
            past_key_values = outputs.past_key_values
            
            generated_ids = self.base_model.generate(
                max_length=100,
                num_beams=4,
                early_stopping=True,
                past_key_values=past_key_values
            )
            
            generated_texts = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            return generated_texts
