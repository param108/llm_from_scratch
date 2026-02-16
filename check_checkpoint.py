"""
Diagnostic script to inspect model checkpoints and vocabulary files.
"""
import sys
import torch
from embeddings import BPE

def check_checkpoint(checkpoint_path, vocab_path):
    """Check checkpoint and vocabulary compatibility."""
    print("=" * 60)
    print("Checkpoint and Vocabulary Diagnostic")
    print("=" * 60)

    # Check vocabulary
    print(f"\n1. Checking vocabulary: {vocab_path}")
    bpe = BPE(vocab_path)
    if not bpe.load():
        print(f"   ERROR: Could not load vocabulary from {vocab_path}")
        return

    num_tokens = len(bpe.lookup)
    model_vocab_size = bpe.get_model_vocab_size()
    print(f"   ✓ Vocabulary loaded successfully")
    print(f"   Number of tokens: {num_tokens}")
    print(f"   Model vocabulary size: {model_vocab_size} (for embedding layer)")

    # Check checkpoint
    print(f"\n2. Checking checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"   ERROR: Could not load checkpoint: {e}")
        return

    print(f"   ✓ Checkpoint loaded successfully")

    # Extract model configuration from checkpoint
    if 'token_embedding.embedding.weight' in checkpoint:
        checkpoint_vocab_size = checkpoint['token_embedding.embedding.weight'].shape[0]
        embedding_dim = checkpoint['token_embedding.embedding.weight'].shape[1]

        print(f"\n3. Model Configuration:")
        print(f"   Vocabulary size: {checkpoint_vocab_size}")
        print(f"   Embedding dimension: {embedding_dim}")

        # Check for position embedding to infer max_seq_len
        if 'position_embedding.weight' in checkpoint:
            max_seq_len = checkpoint['position_embedding.weight'].shape[0]
            print(f"   Max sequence length: {max_seq_len}")

        # Count total parameters
        total_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
        print(f"   Total parameters: {total_params:,}")

        # Check compatibility
        print(f"\n4. Compatibility Check:")
        if checkpoint_vocab_size == model_vocab_size:
            print(f"   ✓ Vocabulary sizes match!")
            print(f"   The checkpoint and vocabulary are compatible.")
        else:
            print(f"   ✗ MISMATCH: Checkpoint vocab ({checkpoint_vocab_size}) != Current model vocab ({model_vocab_size})")
            print(f"   Number of tokens in vocabulary: {num_tokens}")
            print(f"   This will cause index out of bounds errors during training.")
            print(f"\n   Solutions:")
            print(f"   - Use the original vocabulary file that was used to train this checkpoint")
            print(f"   - Or retrain the model from scratch with the current vocabulary")
    else:
        print("   ERROR: Could not find expected tensors in checkpoint")
        print("   Available keys:", list(checkpoint.keys())[:10])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_checkpoint.py <checkpoint_file> <vocab_file>")
        print("\nExample:")
        print("  python check_checkpoint.py gpt2_model.pt vocab.txt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    vocab_path = sys.argv[2]

    check_checkpoint(checkpoint_path, vocab_path)
