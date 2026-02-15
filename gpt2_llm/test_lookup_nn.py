"""Tests for LookupNN embedding layer."""

import pytest
import torch
from gpt2_llm.lookup_nn import LookupNN


class TestLookupNN:
    """Test suite for LookupNN class."""

    def test_init_default_embedding_dim(self):
        """Test initialization with default embedding dimension."""
        vocab_length = 100
        lookup = LookupNN(vocab_length)

        assert lookup.vocab_length == 100
        assert lookup.embedding_dim == 50
        assert lookup.embedding is None

    def test_init_custom_embedding_dim(self):
        """Test initialization with custom embedding dimension."""
        vocab_length = 100
        embedding_dim = 128
        lookup = LookupNN(vocab_length, embedding_dim)

        assert lookup.vocab_length == 100
        assert lookup.embedding_dim == 128
        assert lookup.embedding is None

    def test_create_embedding(self):
        """Test creating the embedding layer."""
        vocab_length = 100
        embedding_dim = 50
        lookup = LookupNN(vocab_length, embedding_dim)

        embedding = lookup.create()

        assert embedding is not None
        assert isinstance(embedding, torch.nn.Embedding)
        assert embedding.num_embeddings == vocab_length
        assert embedding.embedding_dim == embedding_dim

    def test_forward_single_token(self):
        """Test forward pass with a single token ID."""
        vocab_length = 100
        embedding_dim = 50
        lookup = LookupNN(vocab_length, embedding_dim)
        lookup.create()

        token_id = torch.tensor([5])
        output = lookup.forward(token_id)

        assert output.shape == (1, embedding_dim)

    def test_forward_sequence(self):
        """Test forward pass with a sequence of token IDs."""
        vocab_length = 100
        embedding_dim = 50
        lookup = LookupNN(vocab_length, embedding_dim)
        lookup.create()

        token_ids = torch.tensor([1, 5, 10, 15, 20])
        output = lookup.forward(token_ids)

        assert output.shape == (5, embedding_dim)

    def test_forward_batch(self):
        """Test forward pass with a batch of sequences."""
        vocab_length = 100
        embedding_dim = 50
        lookup = LookupNN(vocab_length, embedding_dim)
        lookup.create()

        # Batch of 3 sequences, each with 5 tokens
        token_ids = torch.tensor([
            [1, 5, 10, 15, 20],
            [2, 6, 11, 16, 21],
            [3, 7, 12, 17, 22]
        ])
        output = lookup.forward(token_ids)

        assert output.shape == (3, 5, embedding_dim)

    def test_forward_without_create_raises_error(self):
        """Test that forward raises error if create() not called."""
        vocab_length = 100
        embedding_dim = 50
        lookup = LookupNN(vocab_length, embedding_dim)

        token_ids = torch.tensor([1, 2, 3])

        with pytest.raises(RuntimeError, match="Embedding layer not created"):
            lookup.forward(token_ids)

    def test_get_embedding_matrix(self):
        """Test getting the embedding weight matrix."""
        vocab_length = 100
        embedding_dim = 50
        lookup = LookupNN(vocab_length, embedding_dim)
        lookup.create()

        matrix = lookup.get_embedding_matrix()

        assert matrix.shape == (vocab_length, embedding_dim)
        assert isinstance(matrix, torch.Tensor)

    def test_get_embedding_matrix_without_create_raises_error(self):
        """Test that get_embedding_matrix raises error if create() not called."""
        vocab_length = 100
        embedding_dim = 50
        lookup = LookupNN(vocab_length, embedding_dim)

        with pytest.raises(RuntimeError, match="Embedding layer not created"):
            lookup.get_embedding_matrix()

    def test_save_and_load_embeddings(self, tmp_path):
        """Test saving and loading embeddings."""
        vocab_length = 100
        embedding_dim = 50
        lookup1 = LookupNN(vocab_length, embedding_dim)
        lookup1.create()

        # Get original weights
        original_weights = lookup1.get_embedding_matrix().clone()

        # Save embeddings
        save_path = tmp_path / "embeddings.pt"
        lookup1.save_embeddings(str(save_path))

        # Create new lookup and load embeddings
        lookup2 = LookupNN(vocab_length, embedding_dim)
        lookup2.load_embeddings(str(save_path))

        # Check weights are the same
        loaded_weights = lookup2.get_embedding_matrix()
        assert torch.allclose(original_weights, loaded_weights)

    def test_embeddings_are_trainable(self):
        """Test that embeddings can be trained (gradients flow)."""
        vocab_length = 100
        embedding_dim = 50
        lookup = LookupNN(vocab_length, embedding_dim)
        lookup.create()

        token_ids = torch.tensor([1, 2, 3])
        output = lookup.forward(token_ids)

        # Create a simple loss and backprop
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert lookup.embedding.weight.grad is not None
        assert not torch.all(lookup.embedding.weight.grad == 0)

    def test_different_vocab_sizes(self):
        """Test with various vocabulary sizes."""
        test_cases = [
            (10, 32),
            (100, 50),
            (1000, 128),
            (10000, 256)
        ]

        for vocab_length, embedding_dim in test_cases:
            lookup = LookupNN(vocab_length, embedding_dim)
            lookup.create()

            assert lookup.embedding.num_embeddings == vocab_length
            assert lookup.embedding.embedding_dim == embedding_dim

            # Test forward pass
            token_ids = torch.tensor([0, vocab_length - 1])
            output = lookup.forward(token_ids)
            assert output.shape == (2, embedding_dim)
