import torch
import torch.nn as nn
import numpy as np
from transformers import T5Tokenizer

"""
IMPORTANT NOTE: This implementation strongly recommends using actual VQ-VAE codebook vectors 
rather than one-hot encoded vectors as input for the following reasons:

1. Memory Efficiency: One-hot vectors are extremely sparse and memory-intensive
   For a codebook of 1024 entries, each one-hot vector requires 1024 floats, while
   the actual embedding might only need 512 floats.

2. Computational Efficiency: The projection layer is much smaller with codebook vectors
   - With one-hot: (codebook_size × model_dim) parameters
   - With codebook vectors: (embedding_dim × model_dim) parameters

3. Information Preservation: Using one-hot vectors loses the semantic information already
   captured in the VQ-VAE embeddings, forcing the projection layer to relearn it.

4. Better Generalization: Continuous embedding vectors allow better generalization.

The compare_input_formats() function below demonstrates these advantages.
"""


# Example showing how projection works with an LLM
def projection_example():
    # Parameters
    input_dim = 512  # Dimension of VQ-VAE embeddings
    batch_size = 4
    seq_len = 100

    # Create sample input (simulating VQ-VAE output)
    vq_embeddings = torch.randn(batch_size, seq_len, input_dim)

    # 1. Initialize T5 tokenizer for target text
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # 2. Create embedding projector
    projector = EmbeddingProjector(
        input_dim=input_dim,
        model_dim=512,  # T5-small's hidden dimension
        max_seq_len=200,
    )

    # 3. Project VQ embeddings to T5's embedding space with positional encoding
    projected_embeddings = projector(vq_embeddings)
    print(f"Input shape: {vq_embeddings.shape}")
    print(f"Projected shape: {projected_embeddings.shape}")

    # 4. Example of tokenizing target text
    target_texts = [
        "This is an example sentence.",
        "Another sentence to demonstrate tokenization.",
        "The model learns to map from embeddings to text.",
        "Neural decoding is fascinating.",
    ]

    # Tokenize target texts
    target_encodings = tokenizer(
        target_texts, padding="longest", return_tensors="pt", truncation=True
    )

    print(f"Tokenized target shape: {target_encodings.input_ids.shape}")
    print("Example token IDs:", target_encodings.input_ids[0])
    print("Decoded back:", tokenizer.decode(target_encodings.input_ids[0]))

    # 5. Note: During forward pass, these projected embeddings would be passed to the LLM encoder
    # and the tokenized targets would be used as labels for the decoder


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
            self._create_sinusoidal_encoding(max_seq_len, model_dim),
        )
        self.max_seq_len = max_seq_len

    def _create_sinusoidal_encoding(self, max_seq_len, model_dim):
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * -(np.log(10000.0) / model_dim)
        )

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


# Add this example to show the difference between one-hot and codebook vectors
def compare_input_formats():
    """
    Demonstrates the difference between using one-hot vectors vs. codebook vectors.
    This function clearly shows why codebook vectors are strongly preferred over
    one-hot encoded vectors for input to the model.
    """
    codebook_size = 1024  # Number of entries in the VQ-VAE codebook
    embedding_dim = 512  # Dimension of each codebook vector
    seq_len = 100
    batch_size = 2

    # Create a simulated codebook
    vq_codebook = torch.randn(codebook_size, embedding_dim)

    # Option 1: One-hot encoded input
    # Simulating one-hot vectors pointing to codebook entries
    indices = torch.randint(0, codebook_size, (batch_size, seq_len))
    one_hot = torch.zeros(batch_size, seq_len, codebook_size)
    for b in range(batch_size):
        for s in range(seq_len):
            one_hot[b, s, indices[b, s]] = 1

    # Option 2: Direct codebook vectors
    # Using the indices to look up the actual embeddings from the codebook
    codebook_vectors = torch.zeros(batch_size, seq_len, embedding_dim)
    for b in range(batch_size):
        for s in range(seq_len):
            codebook_vectors[b, s] = vq_codebook[indices[b, s]]

    # Create projectors for both approaches
    one_hot_projector = EmbeddingProjector(
        input_dim=codebook_size, model_dim=512  # Large input dimension for one-hot
    )

    codebook_projector = EmbeddingProjector(
        input_dim=embedding_dim,  # Smaller input dimension for codebook vectors
        model_dim=512,
    )

    # Project both
    one_hot_projected = one_hot_projector(one_hot)
    codebook_projected = codebook_projector(codebook_vectors)

    # Print memory and computation info
    print("\n=== COMPARISON: ONE-HOT vs CODEBOOK VECTORS ===")
    print("This comparison demonstrates why codebook vectors are strongly recommended:")
    print(
        f"One-hot input shape: {one_hot.shape}, "
        f"Memory: {one_hot.element_size() * one_hot.nelement() / 1024 / 1024:.2f} MB"
    )
    print(
        f"Codebook vector input shape: {codebook_vectors.shape}, "
        f"Memory: {codebook_vectors.element_size() * codebook_vectors.nelement() / 1024 / 1024:.2f} MB"
    )

    # Approximate computation cost (parameters × input size)
    one_hot_params = codebook_size * 512  # input_dim × model_dim
    codebook_params = embedding_dim * 512  # input_dim × model_dim

    print(f"One-hot projector parameters: {one_hot_params:,}")
    print(f"Codebook projector parameters: {codebook_params:,}")
    print(
        f"Parameter reduction using codebook vectors: {(1 - codebook_params/one_hot_params)*100:.1f}%"
    )
    print("Additionally, codebook vectors preserve semantic information from VQ-VAE")
    print("=== END COMPARISON ===\n")

    return {
        "one_hot_input": one_hot,
        "codebook_input": codebook_vectors,
        "one_hot_projected": one_hot_projected,
        "codebook_projected": codebook_projected,
    }


def prepare_vqvae_inputs(indices, codebook, use_recommended_method=True):
    """
    Prepares inputs for the MultiModalLLM from VQ-VAE codebook indices

    Args:
        indices: Tensor of shape [batch_size, seq_len] containing codebook indices
        codebook: The VQ-VAE codebook of shape [codebook_size, embedding_dim]
        use_recommended_method: If True (recommended), returns actual codebook vectors
                               If False, returns one-hot encoded vectors (not recommended)

    Returns:
        Tensor of inputs ready for the MultiModalLLM
    """
    batch_size, seq_len = indices.shape
    codebook_size, embedding_dim = codebook.shape

    if use_recommended_method:
        # RECOMMENDED: Convert indices to actual codebook vectors
        embeddings = torch.zeros(batch_size, seq_len, embedding_dim)
        for b in range(batch_size):
            for s in range(seq_len):
                embeddings[b, s] = codebook[indices[b, s]]
        print("Using recommended codebook vectors as input")
        return embeddings
    else:
        # NOT RECOMMENDED: Convert indices to one-hot vectors
        print(
            "WARNING: Using one-hot vectors is not recommended due to memory and "
            "computational inefficiency. Consider using codebook vectors instead."
        )
        one_hot = torch.zeros(batch_size, seq_len, codebook_size)
        for b in range(batch_size):
            for s in range(seq_len):
                one_hot[b, s, indices[b, s]] = 1
        return one_hot


if __name__ == "__main__":
    projection_example()
    print("\n" + "=" * 50 + "\n")
    compare_input_formats()
