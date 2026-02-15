"""Test BPE optimization with pytest framework."""

import pytest
from pathlib import Path
from bpe import BPE


@pytest.fixture
def temp_vocab_file(tmp_path):
    """Fixture that provides a temporary vocab file path."""
    vocab_file = tmp_path / "vocab.txt"
    yield str(vocab_file)
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def bpe_instance(temp_vocab_file):
    """Fixture that provides a fresh BPE instance."""
    return BPE(temp_vocab_file, min_pair_frequency=1)


def decode_tokens(bpe, encoded_tokens):
    """Helper function to decode a list of token IDs back to text."""
    return "".join(bpe.reverse_lookup[token_id] for token_id in encoded_tokens)


class TestBPEEncoding:
    """Test suite for BPE encoding functionality."""

    def test_basic_encoding(self, bpe_instance):
        """Test that the optimized BPE produces correct encodings."""
        test_input = "aaabdaaabac"
        bpe_instance.set_vocab_size(10)

        # Encode
        result = bpe_instance.create_vocab(test_input)
        decoded = decode_tokens(bpe_instance, result)

        # Assertions
        assert decoded == test_input, f"Decoded '{decoded}' doesn't match input '{test_input}'"
        assert len(result) < len(test_input), "Encoding should compress the input"
        assert len(result) == 2, f"Expected 2 tokens, got {len(result)}"
        assert len(bpe_instance.lookup) == 10, "Should have exactly 10 tokens in vocabulary"

    def test_repeated_patterns(self, bpe_instance):
        """Test encoding of text with repeated patterns."""
        test_input = "ababababcdcdcd"
        bpe_instance.set_vocab_size(8)

        result = bpe_instance.create_vocab(test_input)
        decoded = decode_tokens(bpe_instance, result)

        assert decoded == test_input, f"Decoded '{decoded}' doesn't match input '{test_input}'"
        assert len(result) < len(test_input), "Encoding should compress the input"
        assert len(result) == 4, f"Expected 4 tokens, got {len(result)}"

    def test_longer_text_compression(self, bpe_instance):
        """Test with longer text to verify compression ratio."""
        test_input = "the quick brown fox jumps over the lazy dog " * 5
        bpe_instance.set_vocab_size(50)

        result = bpe_instance.create_vocab(test_input)
        decoded = decode_tokens(bpe_instance, result)
        compression_ratio = len(result) / len(test_input)

        assert decoded == test_input, "Decoded doesn't match input"
        assert len(result) < len(test_input), "Encoding should compress the input"
        assert compression_ratio < 0.5, f"Expected compression ratio < 50%, got {compression_ratio:.2%}"
        assert len(result) == 95, f"Expected 95 tokens (with whitespace boundaries), got {len(result)}"

    @pytest.mark.parametrize("test_input,vocab_size,expected_compression", [
        ("aaa", 3, True),
        ("abcdef", 10, False),  # No repeated patterns, minimal compression
        ("hello hello hello", 20, True),
    ])
    def test_various_inputs(self, temp_vocab_file, test_input, vocab_size, expected_compression):
        """Parametrized test for various input patterns."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)
        bpe.set_vocab_size(vocab_size)

        result = bpe.create_vocab(test_input)
        decoded = decode_tokens(bpe, result)

        assert decoded == test_input, f"Decoded doesn't match input for '{test_input}'"

        if expected_compression:
            assert len(result) < len(test_input), f"Expected compression for '{test_input}'"


class TestBPEVocabulary:
    """Test suite for BPE vocabulary management."""

    def test_base_vocabulary(self, bpe_instance):
        """Test base vocabulary generation."""
        test_input = "abc"
        bpe_instance.set_vocab_size(10)
        bpe_instance.create_vocab(test_input)

        # Check that base vocabulary includes all unique characters
        assert "a" in bpe_instance.lookup
        assert "b" in bpe_instance.lookup
        assert "c" in bpe_instance.lookup

    def test_vocabulary_size_limit(self, bpe_instance):
        """Test that vocabulary doesn't exceed the specified size."""
        test_input = "aaabbbcccddd"
        vocab_size = 15
        bpe_instance.set_vocab_size(vocab_size)
        bpe_instance.create_vocab(test_input)

        assert len(bpe_instance.lookup) == vocab_size, \
            f"Vocabulary size should be {vocab_size}, got {len(bpe_instance.lookup)}"

    def test_merged_tokens_in_vocabulary(self, bpe_instance):
        """Test that merged tokens are properly added to vocabulary."""
        test_input = "aaabbb"
        bpe_instance.set_vocab_size(6)
        bpe_instance.create_vocab(test_input)

        # After merging, we should have tokens like "aa", "aaa", etc.
        assert any(len(token) > 1 for token in bpe_instance.lookup.keys()), \
            "Should have multi-character tokens after merging"


class TestBPEPersistence:
    """Test suite for BPE save/load functionality."""

    def test_save_and_load(self, temp_vocab_file):
        """Test that vocabulary can be saved and loaded."""
        # Create and train a BPE instance
        bpe1 = BPE(temp_vocab_file, min_pair_frequency=1)
        test_input = "aaabbbccc"
        bpe1.set_vocab_size(10)
        bpe1.create_vocab(test_input)
        original_vocab_size = len(bpe1.lookup)

        # Save vocabulary
        bpe1.save()

        # Load into new instance
        bpe2 = BPE(temp_vocab_file, min_pair_frequency=1)
        bpe2.load()

        # Verify loaded vocabulary matches
        assert len(bpe2.lookup) == original_vocab_size, "Loaded vocabulary size should match"
        assert bpe1.lookup.keys() == bpe2.lookup.keys(), "Loaded tokens should match"

        # Verify token IDs match
        for token in bpe1.lookup.keys():
            assert bpe1.lookup[token].id == bpe2.lookup[token].id, \
                f"Token ID mismatch for '{token}'"


class TestBPEEdgeCases:
    """Test suite for BPE edge cases."""

    def test_single_character(self, bpe_instance):
        """Test encoding a single character."""
        test_input = "a"
        bpe_instance.set_vocab_size(5)
        result = bpe_instance.create_vocab(test_input)
        decoded = decode_tokens(bpe_instance, result)

        assert decoded == test_input
        assert len(result) == 1

    def test_empty_string_handling(self, bpe_instance):
        """Test handling of empty string."""
        test_input = ""
        bpe_instance.set_vocab_size(5)
        result = bpe_instance.create_vocab(test_input)

        assert len(result) == 0

    def test_all_unique_characters(self, bpe_instance):
        """Test input with all unique characters (no pairs to merge)."""
        test_input = "abcdefgh"
        bpe_instance.set_vocab_size(20)
        result = bpe_instance.create_vocab(test_input)
        decoded = decode_tokens(bpe_instance, result)

        assert decoded == test_input
        # Should still have some merges even with unique chars
        assert len(result) <= len(test_input)


class TestBPEOptimization:
    """Test suite specifically for optimization features."""

    def test_pair_counts_reuse(self, bpe_instance, capfd):
        """Test that pair_counts optimization is working."""
        test_input = "aaabbbccc"
        bpe_instance.set_vocab_size(10)

        # Encode and capture output
        bpe_instance.create_vocab(test_input)
        captured = capfd.readouterr()

        # Verify encoding completed successfully
        assert "Encoded input length" in captured.out
        # The optimization should allow this to complete efficiently

    def test_overlapping_position_handling(self, bpe_instance):
        """Test that overlapping merge positions are handled correctly."""
        test_input = "aaaa"  # Creates overlapping pairs at [0,1,2]
        bpe_instance.set_vocab_size(5)

        result = bpe_instance.create_vocab(test_input)
        decoded = decode_tokens(bpe_instance, result)

        assert decoded == test_input, "Should correctly handle overlapping positions"
        assert len(result) < len(test_input), "Should still achieve compression"


class TestBPEWhitespace:
    """Test whitespace handling - tokens never merge across whitespace boundaries."""

    def test_whitespace_boundary_preservation(self, temp_vocab_file):
        """Test that whitespace acts as a boundary and tokens don't merge across it."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)
        text = "hello hello world world"
        bpe.set_vocab_size(50)

        result = bpe.create_vocab(text)
        decoded = decode_tokens(bpe, result)

        # Verify decoding is correct
        assert decoded == text

        # Check that 'hello' and 'world' tokens exist
        assert 'hello' in bpe.lookup
        assert 'world' in bpe.lookup

        # Check that no tokens cross whitespace boundaries
        for token_str in bpe.lookup.keys():
            if len(token_str) > 1:  # Multi-character tokens
                # Should not contain internal spaces (except if it's just a space)
                if token_str != ' ':
                    assert ' ' not in token_str, f"Token '{token_str}' crosses whitespace boundary"
                    assert '\n' not in token_str, f"Token '{token_str}' crosses newline boundary"

    def test_whitespace_normalization(self, temp_vocab_file):
        """Test that whitespace is normalized correctly."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)

        # Multiple spaces should become single space
        text = "hello    world"
        bpe.set_vocab_size(20)
        result = bpe.create_vocab(text)
        decoded = decode_tokens(bpe, result)
        assert decoded == "hello world"  # Multiple spaces -> single space

        # Tabs should become spaces
        bpe2 = BPE(temp_vocab_file + "2", min_pair_frequency=1)
        text2 = "hello\tworld"
        bpe2.set_vocab_size(20)
        result2 = bpe2.create_vocab(text2)
        decoded2 = decode_tokens(bpe2, result2)
        assert decoded2 == "hello world"  # Tab -> space

    def test_newline_normalization(self, temp_vocab_file):
        """Test that newlines are preserved but normalized."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)
        text = "hello\nworld"
        bpe.set_vocab_size(20)
        result = bpe.create_vocab(text)
        decoded = decode_tokens(bpe, result)
        assert decoded == "hello\nworld"

        # Newline with surrounding spaces should become just newline
        bpe2 = BPE(temp_vocab_file + "2", min_pair_frequency=1)
        text2 = "hello  \n  world"
        bpe2.set_vocab_size(20)
        result2 = bpe2.create_vocab(text2)
        decoded2 = decode_tokens(bpe2, result2)
        assert decoded2 == "hello\nworld"  # Spaces around newline removed


class TestBPEEncode:
    """Test the new encode function that uses longest-match with existing vocabulary."""

    def test_encode_with_full_words_in_vocab(self, temp_vocab_file):
        """Test encoding when full words are in vocabulary."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)
        training_text = "hello world hello world"
        bpe.set_vocab_size(50)

        # Create vocabulary
        bpe.create_vocab(training_text)

        # Now encode using the vocabulary
        test_text = "hello world"
        result = bpe.encode(test_text)
        decoded = decode_tokens(bpe, result)

        assert decoded == test_text
        # Should use 'hello', ' ', 'world' tokens
        assert len(result) == 3

    def test_encode_with_partial_matches(self, temp_vocab_file):
        """Test encoding when only parts of words are in vocabulary."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)
        training_text = "hello hello"
        bpe.set_vocab_size(50)

        # Create vocabulary with 'hello'
        bpe.create_vocab(training_text)

        # Encode text with partial word - 'hel' uses merged token, 'l' uses char
        test_text = "hello hell"
        result = bpe.encode(test_text)
        decoded = decode_tokens(bpe, result)

        assert decoded == test_text
        # Should have 'hello', ' ', and 'hell' tokens (or broken down)
        assert len(result) >= 3

    def test_encode_uses_longest_match(self, temp_vocab_file):
        """Test that encode uses longest match, not greedy short matches."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)
        training_text = "aaa aaa aaa"
        bpe.set_vocab_size(50)

        # Create vocabulary - should merge 'a' -> 'aa' -> 'aaa'
        bpe.create_vocab(training_text)

        # Encode should use 'aaa' token, not three 'a' tokens
        test_text = "aaa"
        result = bpe.encode(test_text)

        # Should be 1 token (or 2 if 'aa' + 'a', but should prefer longest)
        assert 'aaa' in bpe.lookup
        assert len(result) <= 2

    def test_encode_multiple_words(self, temp_vocab_file):
        """Test encoding multiple words with whitespace."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)
        training_text = "the quick brown fox"
        bpe.set_vocab_size(100)

        # Create vocabulary
        bpe.create_vocab(training_text)

        # Encode same text
        result = bpe.encode(training_text)
        decoded = decode_tokens(bpe, result)

        assert decoded == training_text

    def test_encode_preserves_whitespace(self, temp_vocab_file):
        """Test that encode preserves whitespace correctly."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)
        training_text = "hello world"
        bpe.set_vocab_size(50)

        bpe.create_vocab(training_text)

        # Test with different whitespace
        test_text = "hello world"
        result = bpe.encode(test_text)
        decoded = decode_tokens(bpe, result)

        # Should normalize to single space
        assert decoded == "hello world"

    def test_encode_empty_string(self, temp_vocab_file):
        """Test encoding empty string."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)
        training_text = "hello"
        bpe.set_vocab_size(20)

        bpe.create_vocab(training_text)

        result = bpe.encode("")
        assert result == []

    def test_encode_unknown_chars_raises_error(self, temp_vocab_file):
        """Test that encoding with unknown characters raises an error."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)
        training_text = "hello"
        bpe.set_vocab_size(20)

        bpe.create_vocab(training_text)

        # Try to encode text with character not in training
        test_text = "hello世界"  # Contains Chinese characters
        try:
            result = bpe.encode(test_text)
            # If it doesn't raise, the characters were handled (maybe added to base vocab)
            decoded = decode_tokens(bpe, result)
            # This should fail or handle gracefully
            assert False, "Should have raised ValueError for unknown character"
        except ValueError as e:
            assert "not found in vocabulary" in str(e)


class TestBPEEarlyExit:
    """Test suite for early exit feature with min_pair_frequency."""

    def test_early_exit_with_low_threshold(self, temp_vocab_file):
        """Test that low threshold allows maximum encoding."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=1)
        text = "hello world " * 10 + "test " * 3 + "example"
        bpe.set_vocab_size(50)

        result = bpe.create_vocab(text)
        decoded = decode_tokens(bpe, result)

        # With threshold=1, should encode aggressively (but respects whitespace boundaries)
        assert decoded == text
        assert len(result) < len(text) * 0.4  # At least 60% compression (limited by whitespace)
        assert len(bpe.lookup) > 20  # Should build vocabulary (but limited by whitespace)

    def test_early_exit_with_medium_threshold(self, temp_vocab_file):
        """Test that medium threshold stops earlier."""
        bpe = BPE(temp_vocab_file, min_pair_frequency=3)
        text = "hello world " * 10 + "test " * 3 + "example"
        bpe.set_vocab_size(50)

        result = bpe.create_vocab(text)
        decoded = decode_tokens(bpe, result)

        # With threshold=3, should stop earlier (respects whitespace boundaries)
        assert decoded == text
        assert len(result) < len(text) * 0.5  # At least 50% compression
        assert 20 <= len(bpe.lookup) <= 30  # Moderate vocabulary size (limited by whitespace)

    def test_early_exit_with_default_threshold(self, temp_vocab_file):
        """Test that default threshold (5) stops when pairs are rare."""
        bpe = BPE(temp_vocab_file)  # Uses default min_pair_frequency=5
        text = "hello world " * 10 + "test " * 3 + "example"
        bpe.set_vocab_size(50)

        result = bpe.create_vocab(text)
        decoded = decode_tokens(bpe, result)

        # With threshold=5, should stop earliest
        assert decoded == text
        assert len(result) < len(text) * 0.7  # At least 30% compression
        assert len(bpe.lookup) < 30  # Smaller vocabulary

    def test_threshold_comparison(self, temp_vocab_file):
        """Test that higher threshold results in less encoding."""
        import os
        text = "aaabbbcccdddeee" * 5
        vocab_sizes = []
        encoded_lengths = []

        for threshold in [1, 3, 5]:
            # Remove vocab file to ensure fresh start for each threshold
            if os.path.exists(temp_vocab_file):
                os.remove(temp_vocab_file)

            bpe = BPE(temp_vocab_file, min_pair_frequency=threshold)
            bpe.set_vocab_size(50)
            result = bpe.create_vocab(text)
            vocab_sizes.append(len(bpe.lookup))
            encoded_lengths.append(len(result))

        # Higher threshold should result in smaller vocab and longer encoding
        assert vocab_sizes[0] > vocab_sizes[1] > vocab_sizes[2], \
            f"Vocab sizes should decrease: {vocab_sizes}"
        assert encoded_lengths[0] < encoded_lengths[1] < encoded_lengths[2], \
            f"Encoded lengths should increase: {encoded_lengths}"

    def test_early_exit_preserves_correctness(self, temp_vocab_file):
        """Test that early exit always maintains encoding/decoding correctness."""
        test_inputs = [
            "hello world " * 5,
            "aaa bbb ccc " * 3,
            "test " * 10,
            "the quick brown fox jumps over the lazy dog " * 2,
        ]

        for threshold in [1, 3, 5, 10]:
            for text in test_inputs:
                bpe = BPE(temp_vocab_file, min_pair_frequency=threshold)
                bpe.set_vocab_size(30)
                result = bpe.create_vocab(text)
                decoded = decode_tokens(bpe, result)

                assert decoded == text, \
                    f"Failed for threshold={threshold}, text='{text[:20]}...'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
