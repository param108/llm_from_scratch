"""
LookupNN - Neural Network Embedding Layer for BPE Tokens
"""

import torch
import torch.nn as nn


class LookupNN(nn.Module):
    """Neural network embedding layer for token lookup.

    This class creates an embedding layer that maps token IDs to dense vectors.
    """

    def __init__(self, vocab_length, embedding_dim=50):
        """Initialize the LookupNN.

        Args:
            vocab_length: Size of the vocabulary (number of unique tokens)
            embedding_dim: Dimension of the embedding vectors (default: 50)
        """
        super(LookupNN, self).__init__()
        self.vocab_length = vocab_length
        self.embedding_dim = embedding_dim
        self.embedding = None

    def create(self):
        """Create the embedding layer.

        Returns:
            The embedding layer (torch.nn.Embedding)
        """
        self.embedding = nn.Embedding(self.vocab_length, self.embedding_dim)
        return self.embedding

    def forward(self, token_ids):
        """Forward pass through the embedding layer.

        Args:
            token_ids: Tensor of token IDs (shape: [batch_size, seq_length] or [seq_length])

        Returns:
            Embedded tokens (shape: [batch_size, seq_length, embedding_dim] or [seq_length, embedding_dim])
        """
        if self.embedding is None:
            raise RuntimeError("Embedding layer not created. Call create() first.")
        return self.embedding(token_ids)

    def get_embedding_matrix(self):
        """Get the embedding weight matrix.

        Returns:
            The embedding weight matrix (shape: [vocab_length, embedding_dim])
        """
        if self.embedding is None:
            raise RuntimeError("Embedding layer not created. Call create() first.")
        return self.embedding.weight.data

    def save_embeddings(self, filename):
        """Save the embedding weights to a file.

        Args:
            filename: Path to save the embeddings
        """
        if self.embedding is None:
            raise RuntimeError("Embedding layer not created. Call create() first.")
        torch.save(self.embedding.state_dict(), filename)

    def load_embeddings(self, filename):
        """Load embedding weights from a file.

        Args:
            filename: Path to load the embeddings from
        """
        if self.embedding is None:
            self.create()
        self.embedding.load_state_dict(torch.load(filename))
