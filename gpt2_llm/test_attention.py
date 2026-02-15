"""
Tests for the QKV attention mechanisms.
"""

import pytest
import torch
import torch.nn as nn
from gpt2_llm.attention import QKVAttention, CausalQKVAttention, RotaryPositionEmbedding
import tempfile
import os


class TestQKVAttention:
    """Test cases for QKVAttention class."""

    def test_initialization(self):
        """Test that QKVAttention initializes correctly."""
        attn = QKVAttention(embedding_dim=512, num_heads=8)
        assert attn.embedding_dim == 512
        assert attn.num_heads == 8
        assert attn.head_dim == 64
        assert attn.query_proj.in_features == 512
        assert attn.key_proj.in_features == 512
        assert attn.value_proj.in_features == 512

    def test_invalid_num_heads(self):
        """Test that initialization fails when embedding_dim is not divisible by num_heads."""
        with pytest.raises(ValueError, match="must be divisible by"):
            QKVAttention(embedding_dim=512, num_heads=7)

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size, seq_len, embedding_dim = 2, 10, 512
        attn = QKVAttention(embedding_dim=embedding_dim, num_heads=8)
        x = torch.randn(batch_size, seq_len, embedding_dim)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_forward_pass_different_sizes(self):
        """Test forward pass with various input sizes."""
        test_cases = [
            (1, 5, 128, 4),   # (batch, seq, embed, heads)
            (4, 20, 256, 8),
            (2, 50, 512, 16),
        ]

        for batch_size, seq_len, embedding_dim, num_heads in test_cases:
            attn = QKVAttention(embedding_dim=embedding_dim, num_heads=num_heads)
            x = torch.randn(batch_size, seq_len, embedding_dim)
            output = attn(x)
            assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_attention_with_mask(self):
        """Test attention with masking."""
        batch_size, seq_len, embedding_dim = 2, 10, 128
        attn = QKVAttention(embedding_dim=embedding_dim, num_heads=4)
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Create a mask that blocks the last 3 positions
        mask = torch.zeros(seq_len, seq_len)
        mask[:, -3:] = float('-inf')

        output = attn(x, mask=mask)
        assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_gradient_flow(self):
        """Test that gradients flow through the attention mechanism."""
        attn = QKVAttention(embedding_dim=128, num_heads=4)
        x = torch.randn(2, 10, 128, requires_grad=True)

        output = attn(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert attn.query_proj.weight.grad is not None
        assert attn.key_proj.weight.grad is not None
        assert attn.value_proj.weight.grad is not None

    def test_save_and_load(self):
        """Test saving and loading attention weights."""
        attn1 = QKVAttention(embedding_dim=128, num_heads=4)
        attn1.eval()  # Set to eval mode to disable dropout
        x = torch.randn(2, 10, 128)
        output1 = attn1(x)

        # Save weights
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            filepath = f.name

        try:
            attn1.save(filepath)

            # Create new attention and load weights
            attn2 = QKVAttention(embedding_dim=128, num_heads=4)
            attn2.eval()  # Set to eval mode to disable dropout
            attn2.load(filepath)

            # Check that outputs match
            output2 = attn2(x)
            assert torch.allclose(output1, output2)
        finally:
            os.unlink(filepath)

    def test_dropout_training_vs_eval(self):
        """Test that dropout behaves differently in training vs eval mode."""
        torch.manual_seed(42)
        attn = QKVAttention(embedding_dim=128, num_heads=4, dropout=0.5)
        x = torch.randn(2, 10, 128)

        # Training mode
        attn.train()
        output_train1 = attn(x)
        output_train2 = attn(x)

        # Outputs should be different due to dropout
        assert not torch.allclose(output_train1, output_train2)

        # Eval mode
        attn.eval()
        output_eval1 = attn(x)
        output_eval2 = attn(x)

        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval1, output_eval2)


class TestCausalQKVAttention:
    """Test cases for CausalQKVAttention class."""

    def test_initialization(self):
        """Test that CausalQKVAttention initializes correctly."""
        attn = CausalQKVAttention(embedding_dim=512, num_heads=8, max_seq_len=1024)
        assert attn.embedding_dim == 512
        assert attn.num_heads == 8
        assert attn.causal_mask.shape == (1024, 1024)

    def test_causal_mask_structure(self):
        """Test that causal mask has correct structure (lower triangular)."""
        attn = CausalQKVAttention(embedding_dim=128, num_heads=4, max_seq_len=10)
        mask = attn.causal_mask

        # Check diagonal and lower triangle are 0
        for i in range(10):
            for j in range(i + 1):
                assert mask[i, j] == 0.0

        # Check upper triangle is -inf
        for i in range(10):
            for j in range(i + 1, 10):
                assert mask[i, j] == float('-inf')

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size, seq_len, embedding_dim = 2, 10, 512
        attn = CausalQKVAttention(embedding_dim=embedding_dim, num_heads=8)
        x = torch.randn(batch_size, seq_len, embedding_dim)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_causal_masking_effect(self):
        """Test that causal masking prevents attending to future positions."""
        torch.manual_seed(42)
        embedding_dim = 128
        seq_len = 5

        # Create attention layer
        attn = CausalQKVAttention(embedding_dim=embedding_dim, num_heads=4)
        attn.eval()

        # Create input where each position has a distinct pattern
        x = torch.zeros(1, seq_len, embedding_dim)
        for i in range(seq_len):
            x[0, i, :] = i + 1

        output = attn(x)

        # The output at position i should not be affected by positions > i
        # We'll test this by modifying future positions and checking output doesn't change
        x_modified = x.clone()
        x_modified[0, 3:, :] = 999  # Change future positions

        output_modified = attn(x_modified)

        # First 3 positions should be unchanged
        assert torch.allclose(output[0, :3, :], output_modified[0, :3, :], atol=1e-5)

    def test_with_additional_mask(self):
        """Test causal attention with additional masking."""
        batch_size, seq_len, embedding_dim = 2, 10, 128
        attn = CausalQKVAttention(embedding_dim=embedding_dim, num_heads=4)
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Create additional mask (e.g., padding mask)
        additional_mask = torch.zeros(seq_len, seq_len)
        additional_mask[:, -2:] = float('-inf')  # Mask last 2 positions

        output = attn(x, mask=additional_mask)
        assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_variable_sequence_lengths(self):
        """Test that attention works with sequences shorter than max_seq_len."""
        attn = CausalQKVAttention(embedding_dim=128, num_heads=4, max_seq_len=100)

        for seq_len in [5, 10, 20, 50]:
            x = torch.randn(2, seq_len, 128)
            output = attn(x)
            assert output.shape == (2, seq_len, 128)

    def test_save_and_load(self):
        """Test saving and loading causal attention weights."""
        attn1 = CausalQKVAttention(embedding_dim=128, num_heads=4)
        attn1.eval()  # Set to eval mode to disable dropout
        x = torch.randn(2, 10, 128)
        output1 = attn1(x)

        # Save weights
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            filepath = f.name

        try:
            attn1.save(filepath)

            # Create new attention and load weights
            attn2 = CausalQKVAttention(embedding_dim=128, num_heads=4)
            attn2.eval()  # Set to eval mode to disable dropout
            attn2.load(filepath)

            # Check that outputs match
            output2 = attn2(x)
            assert torch.allclose(output1, output2)
        finally:
            os.unlink(filepath)


class TestRotaryPositionEmbedding:
    """Test cases for RotaryPositionEmbedding (RoPE)."""

    def test_initialization(self):
        """Test that RoPE initializes correctly."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=1024)
        assert rope.dim == 64
        assert rope.max_seq_len == 1024
        assert rope.inv_freq.shape == (32,)  # dim // 2

    def test_forward_pass_shape(self):
        """Test that RoPE maintains tensor shapes."""
        rope = RotaryPositionEmbedding(dim=64)
        batch, num_heads, seq_len, head_dim = 2, 8, 10, 64

        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_sequence_lengths(self):
        """Test RoPE with varying sequence lengths."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=1024)

        for seq_len in [5, 10, 50, 100]:
            q = torch.randn(2, 4, seq_len, 64)
            k = torch.randn(2, 4, seq_len, 64)

            q_rot, k_rot = rope(q, k)

            assert q_rot.shape == (2, 4, seq_len, 64)
            assert k_rot.shape == (2, 4, seq_len, 64)

    def test_rotation_changes_vectors(self):
        """Test that RoPE actually modifies Q and K vectors."""
        rope = RotaryPositionEmbedding(dim=64)
        q = torch.randn(2, 4, 10, 64)
        k = torch.randn(2, 4, 10, 64)

        q_rot, k_rot = rope(q, k)

        # Rotated vectors should be different from originals
        assert not torch.allclose(q, q_rot)
        assert not torch.allclose(k, k_rot)

    def test_position_dependence(self):
        """Test that RoPE creates position-dependent representations."""
        rope = RotaryPositionEmbedding(dim=64)

        # Create identical vectors at different positions
        q = torch.ones(1, 1, 5, 64)
        k = torch.ones(1, 1, 5, 64)

        q_rot, k_rot = rope(q, k)

        # After rotation, vectors at different positions should be different
        # Position 0 and position 1 should have different representations
        assert not torch.allclose(q_rot[0, 0, 0, :], q_rot[0, 0, 1, :])
        assert not torch.allclose(k_rot[0, 0, 0, :], k_rot[0, 0, 1, :])

    def test_cache_mechanism(self):
        """Test that RoPE caches cos/sin values for efficiency."""
        rope = RotaryPositionEmbedding(dim=64)
        q = torch.randn(2, 4, 10, 64)
        k = torch.randn(2, 4, 10, 64)

        # First call should create cache
        rope(q, k)
        assert rope._seq_len_cached == 10
        assert rope._cos_cached is not None
        assert rope._sin_cached is not None

        # Second call with same seq_len should reuse cache
        cached_cos = rope._cos_cached
        rope(q, k)
        assert rope._cos_cached is cached_cos  # Same object

        # Different seq_len should update cache
        q2 = torch.randn(2, 4, 20, 64)
        k2 = torch.randn(2, 4, 20, 64)
        rope(q2, k2)
        assert rope._seq_len_cached == 20
        assert rope._cos_cached is not cached_cos  # Different object


class TestQKVAttentionWithRoPE:
    """Test cases for QKVAttention with RoPE enabled."""

    def test_initialization_with_rope(self):
        """Test that attention with RoPE initializes correctly."""
        attn = QKVAttention(embedding_dim=512, num_heads=8, use_rope=True)
        assert attn.use_rope is True
        assert attn.rope is not None
        assert isinstance(attn.rope, RotaryPositionEmbedding)

    def test_forward_pass_with_rope(self):
        """Test forward pass with RoPE enabled."""
        attn = QKVAttention(embedding_dim=128, num_heads=4, use_rope=True)
        x = torch.randn(2, 10, 128)

        output = attn(x)

        assert output.shape == (2, 10, 128)

    def test_rope_vs_no_rope_difference(self):
        """Test that RoPE produces different outputs than standard attention."""
        torch.manual_seed(42)

        # Create two attention layers with same initialization
        attn_no_rope = QKVAttention(embedding_dim=128, num_heads=4, use_rope=False)
        attn_with_rope = QKVAttention(embedding_dim=128, num_heads=4, use_rope=True)

        # Copy weights to ensure same initialization
        attn_with_rope.query_proj.weight.data = attn_no_rope.query_proj.weight.data.clone()
        attn_with_rope.key_proj.weight.data = attn_no_rope.key_proj.weight.data.clone()
        attn_with_rope.value_proj.weight.data = attn_no_rope.value_proj.weight.data.clone()
        attn_with_rope.out_proj.weight.data = attn_no_rope.out_proj.weight.data.clone()

        attn_no_rope.eval()
        attn_with_rope.eval()

        x = torch.randn(2, 10, 128)

        output_no_rope = attn_no_rope(x)
        output_with_rope = attn_with_rope(x)

        # Outputs should be different due to RoPE
        assert not torch.allclose(output_no_rope, output_with_rope, atol=1e-4)

    def test_gradient_flow_with_rope(self):
        """Test that gradients flow through attention with RoPE."""
        attn = QKVAttention(embedding_dim=128, num_heads=4, use_rope=True)
        x = torch.randn(2, 10, 128, requires_grad=True)

        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert attn.query_proj.weight.grad is not None

    def test_save_and_load_with_rope(self):
        """Test saving and loading attention with RoPE."""
        attn1 = QKVAttention(embedding_dim=128, num_heads=4, use_rope=True)
        attn1.eval()
        x = torch.randn(2, 10, 128)
        output1 = attn1(x)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            filepath = f.name

        try:
            attn1.save(filepath)

            attn2 = QKVAttention(embedding_dim=128, num_heads=4, use_rope=True)
            attn2.eval()
            attn2.load(filepath)

            output2 = attn2(x)
            assert torch.allclose(output1, output2)
        finally:
            os.unlink(filepath)


class TestCausalQKVAttentionWithRoPE:
    """Test cases for CausalQKVAttention with RoPE enabled."""

    def test_initialization_with_rope(self):
        """Test that causal attention with RoPE initializes correctly."""
        attn = CausalQKVAttention(embedding_dim=512, num_heads=8, use_rope=True)
        assert attn.use_rope is True
        assert attn.rope is not None

    def test_forward_pass_with_rope(self):
        """Test forward pass with RoPE and causal masking."""
        attn = CausalQKVAttention(embedding_dim=128, num_heads=4, use_rope=True)
        x = torch.randn(2, 10, 128)

        output = attn(x)

        assert output.shape == (2, 10, 128)

    def test_causal_masking_with_rope(self):
        """Test that causal masking still works with RoPE."""
        torch.manual_seed(42)
        attn = CausalQKVAttention(embedding_dim=128, num_heads=4, use_rope=True)
        attn.eval()

        x = torch.zeros(1, 5, 128)
        for i in range(5):
            x[0, i, :] = i + 1

        output = attn(x)

        # Modify future positions
        x_modified = x.clone()
        x_modified[0, 3:, :] = 999

        output_modified = attn(x_modified)

        # First 3 positions should be unchanged (causal masking works)
        assert torch.allclose(output[0, :3, :], output_modified[0, :3, :], atol=1e-5)
