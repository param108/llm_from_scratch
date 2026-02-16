"""
Tests for the Trainer class.
"""

import pytest
import torch
import sys
import os
import tempfile

# Add parent directory to path to import BPE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import BPE
from gpt2_llm.gpt2 import GPT2
from gpt2_llm.trainer import Trainer, TextDataset


class TestTextDataset:
    """Test cases for TextDataset."""

    def test_initialization(self):
        """Test TextDataset initialization."""
        token_ids = list(range(100))
        dataset = TextDataset(token_ids, max_seq_len=10)

        # Should have 100 - 10 = 90 sequences
        assert len(dataset) == 90

    def test_getitem(self):
        """Test getting items from dataset."""
        token_ids = list(range(20))
        dataset = TextDataset(token_ids, max_seq_len=5)

        # Get first sequence
        input_ids, target_ids = dataset[0]

        # Input should be [0, 1, 2, 3, 4]
        assert input_ids.tolist() == [0, 1, 2, 3, 4]
        # Target should be [1, 2, 3, 4, 5]
        assert target_ids.tolist() == [1, 2, 3, 4, 5]

    def test_different_indices(self):
        """Test getting different sequences."""
        token_ids = list(range(20))
        dataset = TextDataset(token_ids, max_seq_len=5)

        # Get sequence at index 5
        input_ids, target_ids = dataset[5]

        # Input should be [5, 6, 7, 8, 9]
        assert input_ids.tolist() == [5, 6, 7, 8, 9]
        # Target should be [6, 7, 8, 9, 10]
        assert target_ids.tolist() == [6, 7, 8, 9, 10]

    def test_tensor_types(self):
        """Test that returned tensors have correct type."""
        token_ids = list(range(20))
        dataset = TextDataset(token_ids, max_seq_len=5)

        input_ids, target_ids = dataset[0]

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(target_ids, torch.Tensor)
        assert input_ids.dtype == torch.long
        assert target_ids.dtype == torch.long


class TestTrainer:
    """Test cases for Trainer."""

    def test_initialization(self):
        """Test Trainer initialization."""
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, embedding_dim=128, num_layers=2, num_heads=4)

        # Create simple BPE
        bpe = BPE("test_vocab.txt")
        # Create minimal vocab
        for i in range(vocab_size):
            bpe.lookup[str(i)] = i
            bpe.reverse_lookup[i] = str(i)

        trainer = Trainer(
            model=model,
            bpe=bpe,
            max_seq_len=10,
            batch_size=2,
            learning_rate=1e-3
        )

        assert trainer.model is model
        assert trainer.bpe is bpe
        assert trainer.max_seq_len == 10
        assert trainer.batch_size == 2
        assert trainer.learning_rate == 1e-3

    def test_device_selection(self):
        """Test that device is selected correctly."""
        vocab_size = 100
        model = GPT2(vocab_size=vocab_size, embedding_dim=64, num_layers=1, num_heads=2)

        bpe = BPE("test_vocab.txt")
        for i in range(vocab_size):
            bpe.lookup[str(i)] = i
            bpe.reverse_lookup[i] = str(i)

        trainer = Trainer(model=model, bpe=bpe)

        # Should have a device set
        assert trainer.device is not None
        assert isinstance(trainer.device, torch.device)

    def test_load_and_prepare_data(self):
        """Test loading and preparing data from file."""
        vocab_size = 100
        model = GPT2(vocab_size=vocab_size, embedding_dim=64, num_layers=1, num_heads=2)

        # Create a simple vocabulary
        bpe = BPE("test_vocab.txt")
        text = "hello world this is a test"
        vocab_size = bpe.get_recommended_vocab_size(text)
        bpe.set_vocab_size(vocab_size)
        bpe.create_vocab(text)

        trainer = Trainer(
            model=model,
            bpe=bpe,
            max_seq_len=10,
            batch_size=2
        )

        # Create temporary file with test data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(text)
            temp_file = f.name

        try:
            # Load and prepare data
            dataloader = trainer.load_and_prepare_data(temp_file)

            # Should return a DataLoader
            assert dataloader is not None
            assert len(dataloader) > 0
        finally:
            os.unlink(temp_file)

    def test_train_epoch(self):
        """Test training for one epoch."""
        vocab_size = 100
        model = GPT2(vocab_size=vocab_size, embedding_dim=64, num_layers=1, num_heads=2)

        # Create simple vocabulary
        bpe = BPE("test_vocab.txt")
        for i in range(vocab_size):
            bpe.lookup[str(i)] = i
            bpe.reverse_lookup[i] = str(i)

        trainer = Trainer(
            model=model,
            bpe=bpe,
            max_seq_len=10,
            batch_size=2,
            learning_rate=1e-3
        )

        # Create simple dataset
        token_ids = [i % vocab_size for i in range(50)]
        dataset = TextDataset(token_ids, max_seq_len=10)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Train for one epoch
        loss = trainer.train_epoch(dataloader)

        # Loss should be a finite number
        assert isinstance(loss, float)
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))

    def test_create_trainer_from_vocab(self):
        """Test creating trainer from vocabulary file."""
        # Create a temporary vocabulary using actual BPE creation
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            vocab_file = f.name
            f.write("hello world test training data for vocabulary creation")

        try:
            # Create vocab from file
            bpe = BPE(vocab_file)
            with open(vocab_file, 'r') as f:
                text = f.read()

            vocab_size = bpe.get_recommended_vocab_size(text)
            bpe.set_vocab_size(vocab_size)
            bpe.create_vocab(text)
            bpe.save()

            # Create trainer from vocab
            trainer = Trainer.create_trainer_from_vocab(
                vocab_path=vocab_file,
                embedding_dim=128,
                num_layers=2,
                num_heads=4
            )

            assert trainer is not None
            assert trainer.model is not None
            assert trainer.bpe is not None
            # Model vocab size should match BPE's model vocab size (last_id)
            assert trainer.model.vocab_size == trainer.bpe.get_model_vocab_size()
        finally:
            if os.path.exists(vocab_file):
                os.unlink(vocab_file)

    def test_continue_training_from_checkpoint(self):
        """Test continuing training from a saved checkpoint."""
        vocab_size = 100

        # Create simple vocabulary
        bpe = BPE("test_vocab.txt")
        for i in range(vocab_size):
            bpe.lookup[str(i)] = i
            bpe.reverse_lookup[i] = str(i)

        # Create and train initial model
        model1 = GPT2(vocab_size=vocab_size, embedding_dim=64, num_layers=1, num_heads=2)
        trainer1 = Trainer(
            model=model1,
            bpe=bpe,
            max_seq_len=10,
            batch_size=2,
            learning_rate=1e-3
        )

        # Train for one epoch
        token_ids = [i % vocab_size for i in range(50)]
        dataset = TextDataset(token_ids, max_seq_len=10)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        loss1 = trainer1.train_epoch(dataloader)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            checkpoint_file = f.name

        try:
            model1.save(checkpoint_file)

            # Load checkpoint and continue training
            model2 = GPT2(vocab_size=vocab_size, embedding_dim=64, num_layers=1, num_heads=2)
            model2.load(checkpoint_file)

            trainer2 = Trainer(
                model=model2,
                bpe=bpe,
                max_seq_len=10,
                batch_size=2,
                learning_rate=1e-3
            )

            # Model should be properly loaded
            assert trainer2.model is not None

            # Continue training
            loss2 = trainer2.train_epoch(dataloader)

            # Loss should be a valid number
            assert isinstance(loss2, float)
            assert loss2 > 0
        finally:
            if os.path.exists(checkpoint_file):
                os.unlink(checkpoint_file)
