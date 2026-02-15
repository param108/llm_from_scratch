"""
Trainer for GPT-2 language model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Add parent directory to path to import BPE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bpe import BPE
from .gpt2 import GPT2


class TextDataset(Dataset):
    """
    Dataset for language modeling.

    Creates sequences of tokens for next-token prediction.
    Each sequence has max_seq_len tokens, and the target is to predict
    the next token at each position.
    """

    def __init__(self, token_ids, max_seq_len=1024):
        """
        Args:
            token_ids: List of token IDs
            max_seq_len: Maximum sequence length (context window)
        """
        self.token_ids = token_ids
        self.max_seq_len = max_seq_len

        # Calculate number of sequences we can create
        # We need max_seq_len + 1 tokens (input + target)
        self.num_sequences = max(0, len(token_ids) - max_seq_len)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Returns:
            input_ids: Tensor of shape (max_seq_len,)
            target_ids: Tensor of shape (max_seq_len,)
        """
        # Get sequence starting at idx
        start_idx = idx
        end_idx = start_idx + self.max_seq_len + 1

        sequence = self.token_ids[start_idx:end_idx]

        # Input is all but last token
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        # Target is all but first token (shifted by 1)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)

        return input_ids, target_ids


class Trainer:
    """
    Trainer for GPT-2 language model.

    Handles data loading, training loop, and model saving.
    """

    def __init__(
        self,
        model,
        bpe,
        max_seq_len=1024,
        batch_size=4,
        learning_rate=3e-4,
        device=None
    ):
        """
        Args:
            model: GPT2 model instance
            bpe: BPE tokenizer instance
            max_seq_len: Maximum sequence length (default: 1024)
            batch_size: Training batch size (default: 4)
            learning_rate: Learning rate for optimizer (default: 3e-4)
            device: Device to train on (default: auto-detect)
        """
        self.model = model
        self.bpe = bpe
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def load_and_prepare_data(self, filepath):
        """
        Load text from file and prepare for training.

        Args:
            filepath: Path to text file

        Returns:
            DataLoader for training
        """
        print(f"Loading data from {filepath}...")

        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"Text length: {len(text)} characters")

        # Normalize whitespace using BPE
        text = self.bpe.normalize_whitespace(text)
        print(f"After normalization: {len(text)} characters")

        # Encode to tokens
        print("Encoding text...")
        token_ids = self.bpe.encode(text)
        print(f"Number of tokens: {len(token_ids)}")

        # Validate token IDs
        model_vocab_size = self.bpe.get_model_vocab_size()
        max_token_id = max(token_ids) if token_ids else 0
        min_token_id = min(token_ids) if token_ids else 0

        print(f"Token ID range: [{min_token_id}, {max_token_id}]")
        print(f"Model vocabulary size: {model_vocab_size}")

        if max_token_id >= model_vocab_size:
            raise ValueError(
                f"Token ID {max_token_id} exceeds model vocabulary size {model_vocab_size}. "
                f"This indicates a mismatch between the vocabulary and the model. "
                f"Vocabulary has {len(self.bpe.lookup)} tokens, max ID is {self.bpe.last_id - 1}."
            )

        if min_token_id < 0:
            raise ValueError(f"Invalid negative token ID {min_token_id} found in encoded data.")

        # Create dataset
        dataset = TextDataset(token_ids, max_seq_len=self.max_seq_len)
        print(f"Number of training sequences: {len(dataset)}")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )

        return dataloader

    def train_epoch(self, dataloader):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader with training data

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            logits = self.model(input_ids, return_logits=True)

            # Reshape for loss computation
            # logits: (batch, seq_len, vocab_size)
            # target_ids: (batch, seq_len)
            batch_size, seq_len, vocab_size = logits.shape

            # Flatten to (batch * seq_len, vocab_size) and (batch * seq_len,)
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = target_ids.view(-1)

            # Compute loss
            loss = self.criterion(logits_flat, targets_flat)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def train(self, filepath, num_epochs=1, save_path="gpt2_model.pt"):
        """
        Train the model on data from file.

        Args:
            filepath: Path to training data file
            num_epochs: Number of training epochs (default: 1)
            save_path: Path to save trained model (default: "gpt2_model.pt")
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")

        # Load and prepare data
        dataloader = self.load_and_prepare_data(filepath)

        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            avg_loss = self.train_epoch(dataloader)

            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        # Save model
        print(f"\nSaving model to {save_path}...")
        self.model.save(save_path)
        print("Training completed!")

    @staticmethod
    def create_trainer_from_vocab(
        vocab_path,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=1024,
        batch_size=4,
        learning_rate=3e-4
    ):
        """
        Create a trainer with a new model and existing vocabulary.

        Args:
            vocab_path: Path to BPE vocabulary file
            embedding_dim: Model embedding dimension (default: 768)
            num_layers: Number of decoder layers (default: 12)
            num_heads: Number of attention heads (default: 12)
            max_seq_len: Maximum sequence length (default: 1024)
            batch_size: Training batch size (default: 4)
            learning_rate: Learning rate (default: 3e-4)

        Returns:
            Trainer instance
        """
        # Load BPE
        bpe = BPE(vocab_path)
        if not bpe.load():
            raise ValueError(f"Could not load vocabulary from {vocab_path}")

        vocab_size = bpe.get_model_vocab_size()
        print(f"Loaded vocabulary with {len(bpe.lookup)} tokens (vocab_size={vocab_size})")

        # Create model
        model = GPT2(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            bpe=bpe,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        return trainer
