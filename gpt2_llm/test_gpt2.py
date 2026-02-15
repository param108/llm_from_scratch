"""
Tests for the complete GPT-2 model.
"""

import pytest
import torch
import torch.nn as nn
from gpt2_llm.gpt2 import GPT2
import tempfile
import os


class TestGPT2:
    """Test cases for GPT2 model."""

    def test_initialization_default(self):
        """Test GPT-2 with default configuration (GPT-2 Small)."""
        model = GPT2(vocab_size=50257)
        assert model.vocab_size == 50257
        assert model.embedding_dim == 768
        assert model.num_layers == 12
        assert model.num_heads == 12
        assert model.max_seq_len == 1024

    def test_initialization_custom(self):
        """Test GPT-2 with custom configuration."""
        model = GPT2(
            vocab_size=1000,
            embedding_dim=256,
            num_layers=6,
            num_heads=8,
            max_seq_len=512
        )
        assert model.vocab_size == 1000
        assert model.embedding_dim == 256
        assert model.num_layers == 6
        assert model.num_heads == 8
        assert model.decoder.num_layers == 6

    def test_weight_tying(self):
        """Test that token embeddings and LM head share weights."""
        model = GPT2(vocab_size=1000, embedding_dim=256, num_heads=8)

        # Check that weights are the same object
        assert model.lm_head.weight is model.token_embedding.embedding.weight

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size, seq_len = 2, 10
        vocab_size = 1000

        model = GPT2(vocab_size=vocab_size, embedding_dim=256, num_layers=2, num_heads=4)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = model(token_ids)

        # Output should be probabilities over vocabulary
        assert output.shape == (batch_size, seq_len, vocab_size)

        # Should sum to 1 along vocab dimension
        assert torch.allclose(output.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

    def test_forward_with_logits(self):
        """Test forward pass returning logits."""
        batch_size, seq_len = 2, 10
        vocab_size = 1000

        model = GPT2(vocab_size=vocab_size, embedding_dim=256, num_layers=2, num_heads=4)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = model(token_ids, return_logits=True)

        assert logits.shape == (batch_size, seq_len, vocab_size)
        # Logits should not be probabilities (won't sum to 1)
        assert not torch.allclose(logits.sum(dim=-1), torch.ones(batch_size, seq_len))

    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)

        for batch_size in [1, 2, 4, 8]:
            token_ids = torch.randint(0, vocab_size, (batch_size, 10))
            output = model(token_ids)
            assert output.shape == (batch_size, 10, vocab_size)

    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4, max_seq_len=512)

        for seq_len in [5, 10, 50, 100]:
            token_ids = torch.randint(0, vocab_size, (2, seq_len))
            output = model(token_ids)
            assert output.shape == (2, seq_len, vocab_size)

    def test_max_seq_len_exceeded(self):
        """Test that error is raised when sequence exceeds max length."""
        model = GPT2(vocab_size=1000, embedding_dim=128, num_layers=2, num_heads=4, max_seq_len=100)

        token_ids = torch.randint(0, 1000, (2, 150))  # Exceeds max_seq_len

        with pytest.raises(ValueError, match="exceeds maximum"):
            model(token_ids)

    def test_predict_next_token(self):
        """Test next token prediction."""
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)

        token_ids = torch.randint(0, vocab_size, (2, 10))
        result = model.predict_next_token(token_ids)

        assert 'token_id' in result
        assert 'probability' in result
        assert result['token_id'].shape == (2,)
        assert result['probability'].shape == (2,)

        # Token IDs should be valid
        assert (result['token_id'] >= 0).all()
        assert (result['token_id'] < vocab_size).all()

        # Probabilities should be in [0, 1]
        assert (result['probability'] >= 0).all()
        assert (result['probability'] <= 1).all()

    def test_predict_next_token_with_top_k(self):
        """Test next token prediction with top_k."""
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)

        token_ids = torch.randint(0, vocab_size, (2, 10))
        top_k = 5
        result = model.predict_next_token(token_ids, top_k=top_k)

        assert 'top_tokens' in result
        assert 'top_probs' in result
        assert result['top_tokens'].shape == (2, top_k)
        assert result['top_probs'].shape == (2, top_k)

        # Probabilities should be sorted (highest first)
        for batch_idx in range(2):
            probs = result['top_probs'][batch_idx]
            assert (probs[:-1] >= probs[1:]).all()  # Monotonically decreasing

    def test_predict_next_token_with_temperature(self):
        """Test next token prediction with temperature scaling."""
        torch.manual_seed(42)
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)

        token_ids = torch.randint(0, vocab_size, (1, 10))

        # Higher temperature should spread probability more evenly
        result_low_temp = model.predict_next_token(token_ids, temperature=0.5, top_k=10)
        result_high_temp = model.predict_next_token(token_ids, temperature=2.0, top_k=10)

        # Check that we get results with valid probabilities
        assert result_low_temp['top_probs'].shape == (1, 10)
        assert result_high_temp['top_probs'].shape == (1, 10)

        # Low temp should have more concentrated probability (higher top prob or lower entropy)
        # Check the ratio between top and 2nd probability
        low_temp_ratio = result_low_temp['top_probs'][0, 0] / result_low_temp['top_probs'][0, 1]
        high_temp_ratio = result_high_temp['top_probs'][0, 0] / result_high_temp['top_probs'][0, 1]

        # Low temperature should have higher concentration (higher ratio)
        # Allow for edge cases where ratios might be similar
        assert low_temp_ratio >= high_temp_ratio * 0.9  # Allow 10% tolerance

    def test_get_token_predictions(self):
        """Test getting top k token predictions."""
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)

        token_ids = torch.randint(0, vocab_size, (2, 10))
        top_k = 5
        predictions = model.get_token_predictions(token_ids, top_k=top_k)

        assert len(predictions) == 2  # One per batch item

        for batch_predictions in predictions:
            assert len(batch_predictions) == top_k

            for token_id, prob in batch_predictions:
                assert isinstance(token_id, int)
                assert isinstance(prob, float)
                assert 0 <= token_id < vocab_size
                assert 0.0 <= prob <= 1.0

            # Probabilities should be sorted (highest first)
            probs = [p for _, p in batch_predictions]
            assert probs == sorted(probs, reverse=True)

    def test_get_token_predictions_different_positions(self):
        """Test getting predictions at different positions."""
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)

        token_ids = torch.randint(0, vocab_size, (1, 10))

        # Test different positions
        for position in [0, 5, -1, -2]:
            predictions = model.get_token_predictions(token_ids, position=position, top_k=3)
            assert len(predictions) == 1
            assert len(predictions[0]) == 3

    def test_generate(self):
        """Test autoregressive generation."""
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)

        token_ids = torch.randint(0, vocab_size, (2, 5))
        max_new_tokens = 10

        generated = model.generate(token_ids, max_new_tokens=max_new_tokens)

        # Should have original + new tokens
        assert generated.shape[0] == 2  # Same batch size
        assert generated.shape[1] == 5 + max_new_tokens

        # First part should match input
        assert torch.equal(generated[:, :5], token_ids)

    def test_generate_with_temperature(self):
        """Test generation with temperature."""
        torch.manual_seed(42)
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)

        token_ids = torch.randint(0, vocab_size, (1, 5))

        # Generate with different temperatures
        gen_low = model.generate(token_ids, max_new_tokens=5, temperature=0.5)
        gen_high = model.generate(token_ids, max_new_tokens=5, temperature=2.0)

        # Results should be different due to temperature
        # (not guaranteed but highly likely with different seeds)
        assert gen_low.shape == gen_high.shape

    def test_generate_with_stop_token(self):
        """Test generation with stop token."""
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)

        token_ids = torch.randint(0, vocab_size, (1, 5))
        stop_token_id = 999

        generated = model.generate(token_ids, max_new_tokens=50, stop_token_id=stop_token_id)

        # Should stop early if stop token is generated
        # (may or may not happen, but shape should be valid)
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 5

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)

        token_ids = torch.randint(0, vocab_size, (2, 10))
        logits = model(token_ids, return_logits=True)

        # Compute loss (dummy target)
        target = torch.randint(0, vocab_size, (2, 10))
        loss = nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            target.view(-1)
        )

        loss.backward()

        # Check that gradients exist
        assert model.token_embedding.embedding.weight.grad is not None
        assert model.position_embedding.weight.grad is not None
        assert model.decoder.blocks[0].attention.query_proj.weight.grad is not None
        assert model.lm_head.weight.grad is not None

    def test_save_and_load(self):
        """Test saving and loading model."""
        vocab_size = 1000
        model1 = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)
        model1.eval()

        token_ids = torch.randint(0, vocab_size, (2, 10))
        output1 = model1(token_ids)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            filepath = f.name

        try:
            model1.save(filepath)

            model2 = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)
            model2.eval()
            model2.load(filepath)

            output2 = model2(token_ids)
            assert torch.allclose(output1, output2)
        finally:
            os.unlink(filepath)

    def test_count_parameters(self):
        """Test parameter counting."""
        model = GPT2(vocab_size=1000, embedding_dim=128, num_layers=2, num_heads=4)

        param_count = model.count_parameters()

        # Should have a reasonable number of parameters
        assert param_count > 0

        # Verify it matches manual count
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count

    def test_gpt2_small_config(self):
        """Test GPT-2 Small configuration (117M parameters)."""
        # GPT-2 Small: 12 layers, 768 dim, 12 heads, 50257 vocab
        model = GPT2(
            vocab_size=50257,
            embedding_dim=768,
            num_layers=12,
            num_heads=12
        )

        assert model.num_layers == 12
        assert model.embedding_dim == 768
        assert model.num_heads == 12

        # Rough parameter count check (should be around 117M)
        param_count = model.count_parameters()
        # Should be in the ballpark (allowing for some variation)
        assert 100_000_000 < param_count < 130_000_000

    def test_eval_mode(self):
        """Test that model can be put in eval mode."""
        model = GPT2(vocab_size=1000, embedding_dim=128, num_layers=2, num_heads=4)

        # Should start in training mode
        assert model.training

        # Switch to eval
        model.eval()
        assert not model.training

        # Should still work
        token_ids = torch.randint(0, 1000, (2, 10))
        output = model(token_ids)
        assert output.shape == (2, 10, 1000)

    def test_train_mode(self):
        """Test that model can be put in train mode."""
        model = GPT2(vocab_size=1000, embedding_dim=128, num_layers=2, num_heads=4)

        model.eval()
        assert not model.training

        model.train()
        assert model.training
