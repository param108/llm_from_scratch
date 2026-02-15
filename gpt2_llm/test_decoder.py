"""
Tests for GPT-2 decoder components.
"""

import pytest
import torch
import torch.nn as nn
from gpt2_llm.decoder import FeedForward, GPT2DecoderBlock, GPT2Decoder
import tempfile
import os


class TestFeedForward:
    """Test cases for FeedForward (MLP) network."""

    def test_initialization_default(self):
        """Test FeedForward with default hidden dimension (4x expansion)."""
        ffn = FeedForward(embedding_dim=512)
        assert ffn.embedding_dim == 512
        assert ffn.hidden_dim == 2048  # 4 * 512
        assert ffn.fc1.in_features == 512
        assert ffn.fc1.out_features == 2048
        assert ffn.fc2.in_features == 2048
        assert ffn.fc2.out_features == 512

    def test_initialization_custom(self):
        """Test FeedForward with custom hidden dimension."""
        ffn = FeedForward(embedding_dim=512, hidden_dim=1024)
        assert ffn.embedding_dim == 512
        assert ffn.hidden_dim == 1024
        assert ffn.fc1.out_features == 1024
        assert ffn.fc2.in_features == 1024

    def test_forward_pass_shape(self):
        """Test that forward pass maintains correct shape."""
        batch_size, seq_len, embedding_dim = 2, 10, 512
        ffn = FeedForward(embedding_dim=embedding_dim)
        x = torch.randn(batch_size, seq_len, embedding_dim)

        output = ffn(x)

        assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_forward_pass_different_sizes(self):
        """Test forward pass with various input sizes."""
        test_cases = [
            (1, 5, 128),
            (4, 20, 256),
            (2, 50, 768),
        ]

        for batch_size, seq_len, embedding_dim in test_cases:
            ffn = FeedForward(embedding_dim=embedding_dim)
            x = torch.randn(batch_size, seq_len, embedding_dim)
            output = ffn(x)
            assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_activation_is_gelu(self):
        """Test that GELU activation is used."""
        ffn = FeedForward(embedding_dim=128)
        assert isinstance(ffn.activation, nn.GELU)

    def test_gradient_flow(self):
        """Test that gradients flow through FFN."""
        ffn = FeedForward(embedding_dim=128)
        x = torch.randn(2, 10, 128, requires_grad=True)

        output = ffn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert ffn.fc1.weight.grad is not None
        assert ffn.fc2.weight.grad is not None

    def test_dropout_training_vs_eval(self):
        """Test that dropout behaves differently in training vs eval."""
        torch.manual_seed(42)
        ffn = FeedForward(embedding_dim=128, dropout=0.5)
        x = torch.randn(2, 10, 128)

        # Training mode
        ffn.train()
        output_train1 = ffn(x)
        output_train2 = ffn(x)
        assert not torch.allclose(output_train1, output_train2)

        # Eval mode
        ffn.eval()
        output_eval1 = ffn(x)
        output_eval2 = ffn(x)
        assert torch.allclose(output_eval1, output_eval2)


class TestGPT2DecoderBlock:
    """Test cases for GPT2DecoderBlock."""

    def test_initialization(self):
        """Test decoder block initialization."""
        block = GPT2DecoderBlock(embedding_dim=512, num_heads=8)
        assert block.embedding_dim == 512
        assert block.num_heads == 8
        assert isinstance(block.ln1, nn.LayerNorm)
        assert isinstance(block.ln2, nn.LayerNorm)
        assert block.attention is not None
        assert block.ffn is not None

    def test_forward_pass_shape(self):
        """Test that forward pass maintains correct shape."""
        batch_size, seq_len, embedding_dim = 2, 10, 512
        block = GPT2DecoderBlock(embedding_dim=embedding_dim, num_heads=8)
        x = torch.randn(batch_size, seq_len, embedding_dim)

        output = block(x)

        assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_forward_pass_different_sizes(self):
        """Test forward pass with various configurations."""
        test_cases = [
            (1, 5, 128, 4),    # (batch, seq, embed, heads)
            (4, 20, 256, 8),
            (2, 50, 512, 16),
        ]

        for batch_size, seq_len, embedding_dim, num_heads in test_cases:
            block = GPT2DecoderBlock(embedding_dim=embedding_dim, num_heads=num_heads)
            x = torch.randn(batch_size, seq_len, embedding_dim)
            output = block(x)
            assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_residual_connections(self):
        """Test that residual connections are working."""
        torch.manual_seed(42)
        block = GPT2DecoderBlock(embedding_dim=128, num_heads=4, dropout=0.0)
        block.eval()

        x = torch.randn(1, 10, 128)
        output = block(x)

        # Output should be different from input (not just identity)
        assert not torch.allclose(x, output)

        # But should still have some relationship (residual adds to output)
        # The output magnitude should be related to input magnitude
        assert output.abs().mean() > 0

    def test_with_rope(self):
        """Test decoder block with RoPE enabled."""
        block = GPT2DecoderBlock(
            embedding_dim=128,
            num_heads=4,
            use_rope=True
        )
        x = torch.randn(2, 10, 128)

        output = block(x)

        assert output.shape == (2, 10, 128)
        assert block.attention.use_rope is True

    def test_gradient_flow(self):
        """Test that gradients flow through the entire block."""
        block = GPT2DecoderBlock(embedding_dim=128, num_heads=4)
        x = torch.randn(2, 10, 128, requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert block.attention.query_proj.weight.grad is not None
        assert block.ffn.fc1.weight.grad is not None
        assert block.ln1.weight.grad is not None
        assert block.ln2.weight.grad is not None

    def test_save_and_load(self):
        """Test saving and loading decoder block."""
        block1 = GPT2DecoderBlock(embedding_dim=128, num_heads=4)
        block1.eval()
        x = torch.randn(2, 10, 128)
        output1 = block1(x)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            filepath = f.name

        try:
            block1.save(filepath)

            block2 = GPT2DecoderBlock(embedding_dim=128, num_heads=4)
            block2.eval()
            block2.load(filepath)

            output2 = block2(x)
            assert torch.allclose(output1, output2)
        finally:
            os.unlink(filepath)

    def test_causal_masking(self):
        """Test that causal masking is applied correctly."""
        torch.manual_seed(42)
        block = GPT2DecoderBlock(embedding_dim=128, num_heads=4)
        block.eval()

        # Create input where each position has distinct values
        x = torch.zeros(1, 5, 128)
        for i in range(5):
            x[0, i, :] = i + 1

        output = block(x)

        # Modify future positions
        x_modified = x.clone()
        x_modified[0, 3:, :] = 999

        output_modified = block(x_modified)

        # First 3 positions should be unchanged due to causal masking
        assert torch.allclose(output[0, :3, :], output_modified[0, :3, :], atol=1e-5)


class TestGPT2Decoder:
    """Test cases for full GPT2Decoder."""

    def test_initialization(self):
        """Test decoder initialization."""
        decoder = GPT2Decoder(
            num_layers=6,
            embedding_dim=512,
            num_heads=8
        )
        assert decoder.num_layers == 6
        assert decoder.embedding_dim == 512
        assert decoder.num_heads == 8
        assert len(decoder.blocks) == 6
        assert isinstance(decoder.ln_final, nn.LayerNorm)

    def test_forward_pass_shape(self):
        """Test that forward pass maintains correct shape."""
        batch_size, seq_len, embedding_dim = 2, 10, 256
        decoder = GPT2Decoder(
            num_layers=4,
            embedding_dim=embedding_dim,
            num_heads=4
        )
        x = torch.randn(batch_size, seq_len, embedding_dim)

        output = decoder(x)

        assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_different_layer_counts(self):
        """Test decoder with different numbers of layers."""
        for num_layers in [1, 2, 6, 12]:
            decoder = GPT2Decoder(
                num_layers=num_layers,
                embedding_dim=128,
                num_heads=4
            )
            assert len(decoder.blocks) == num_layers

            x = torch.randn(2, 10, 128)
            output = decoder(x)
            assert output.shape == (2, 10, 128)

    def test_with_rope(self):
        """Test full decoder with RoPE enabled."""
        decoder = GPT2Decoder(
            num_layers=3,
            embedding_dim=128,
            num_heads=4,
            use_rope=True
        )
        x = torch.randn(2, 10, 128)

        output = decoder(x)

        assert output.shape == (2, 10, 128)
        # Check that all blocks have RoPE enabled
        for block in decoder.blocks:
            assert block.attention.use_rope is True

    def test_gradient_flow(self):
        """Test that gradients flow through all layers."""
        decoder = GPT2Decoder(
            num_layers=3,
            embedding_dim=128,
            num_heads=4
        )
        x = torch.randn(2, 10, 128, requires_grad=True)

        output = decoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        # Check gradients in first and last blocks
        assert decoder.blocks[0].attention.query_proj.weight.grad is not None
        assert decoder.blocks[-1].ffn.fc1.weight.grad is not None
        assert decoder.ln_final.weight.grad is not None

    def test_save_and_load(self):
        """Test saving and loading full decoder."""
        decoder1 = GPT2Decoder(
            num_layers=2,
            embedding_dim=128,
            num_heads=4
        )
        decoder1.eval()
        x = torch.randn(2, 10, 128)
        output1 = decoder1(x)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            filepath = f.name

        try:
            decoder1.save(filepath)

            decoder2 = GPT2Decoder(
                num_layers=2,
                embedding_dim=128,
                num_heads=4
            )
            decoder2.eval()
            decoder2.load(filepath)

            output2 = decoder2(x)
            assert torch.allclose(output1, output2)
        finally:
            os.unlink(filepath)

    def test_causal_behavior(self):
        """Test that decoder maintains causal behavior through all layers."""
        torch.manual_seed(42)
        decoder = GPT2Decoder(
            num_layers=3,
            embedding_dim=128,
            num_heads=4
        )
        decoder.eval()

        x = torch.zeros(1, 5, 128)
        for i in range(5):
            x[0, i, :] = i + 1

        output = decoder(x)

        # Modify future positions
        x_modified = x.clone()
        x_modified[0, 3:, :] = 999

        output_modified = decoder(x_modified)

        # First 3 positions should be unchanged
        assert torch.allclose(output[0, :3, :], output_modified[0, :3, :], atol=1e-5)

    def test_custom_ffn_dim(self):
        """Test decoder with custom FFN hidden dimension."""
        decoder = GPT2Decoder(
            num_layers=2,
            embedding_dim=128,
            num_heads=4,
            ffn_hidden_dim=256  # Custom instead of 4x
        )

        # Check that FFN has correct dimension
        assert decoder.blocks[0].ffn.hidden_dim == 256

        x = torch.randn(2, 10, 128)
        output = decoder(x)
        assert output.shape == (2, 10, 128)
