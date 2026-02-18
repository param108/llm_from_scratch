import sys
import json
import torch
from embeddings import BPE
from gpt2_llm.trainer import Trainer
from gpt2_llm.gpt2 import GPT2

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [args...]")
        print("Commands:")
        print("  tokenize <file>                  - Create vocabulary from file")
        print("  encode <text> [vocab_file]       - Encode text using vocabulary")
        print("  decode <json_array> [vocab_file] - Decode token IDs to text")
        print("  train <file> [vocab_file] [checkpoint] - Train GPT-2 model on file")
        print("  predict <prompt> <num_tokens> [model_file] [vocab_file] - Generate text")
        sys.exit(1)

    command = sys.argv[1]

    if command == "tokenize":
        if len(sys.argv) < 3:
            print("Usage: python main.py tokenize <file>")
            sys.exit(1)

        filename = sys.argv[2]

        # Read the file
        with open(filename, 'r') as f:
            text = f.read()

        # Create BPE instance and create vocabulary
        bpe = BPE("vocab.txt")
        vocab_size = bpe.get_recommended_vocab_size(text)
        print("Recommended vocabulary size:", vocab_size)
        bpe.set_vocab_size(vocab_size)
        bpe.create_vocab(text)
        bpe.save()

    elif command == "encode":
        if len(sys.argv) < 3:
            print("Usage: python main.py encode <text> [vocab_file]")
            sys.exit(1)

        text = sys.argv[2]
        vocab_file = sys.argv[3] if len(sys.argv) > 3 else "vocab.txt"

        # Create BPE instance and load vocabulary
        bpe = BPE(vocab_file)
        loaded = bpe.load()

        if not loaded:
            print(f"Error: Vocabulary file '{vocab_file}' not found.", file=sys.stderr)
            print("Please create a vocabulary first using the 'tokenize' command.", file=sys.stderr)
            sys.exit(1)

        # Encode the text
        try:
            token_ids = bpe.encode(text)
            # Output as JSON array
            print(json.dumps(token_ids))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif command == "decode":
        if len(sys.argv) < 3:
            print("Usage: python main.py decode <json_array> [vocab_file]")
            sys.exit(1)

        json_str = sys.argv[2]
        vocab_file = sys.argv[3] if len(sys.argv) > 3 else "vocab.txt"

        # Create BPE instance and load vocabulary
        bpe = BPE(vocab_file)
        loaded = bpe.load()

        if not loaded:
            print(f"Error: Vocabulary file '{vocab_file}' not found.", file=sys.stderr)
            print("Please create a vocabulary first using the 'tokenize' command.", file=sys.stderr)
            sys.exit(1)

        # Parse JSON array
        try:
            token_ids = json.loads(json_str)
            if not isinstance(token_ids, list):
                print("Error: Input must be a JSON array of integers.", file=sys.stderr)
                sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)

        # Decode the token IDs to list of strings
        try:
            decoded_tokens = []
            for token_id in token_ids:
                if token_id in bpe.reverse_lookup:
                    decoded_tokens.append(bpe.reverse_lookup[token_id])
                else:
                    print(f"Error: Token ID {token_id} not found in vocabulary.", file=sys.stderr)
                    sys.exit(1)
            # Output as JSON array of strings
            print(json.dumps(decoded_tokens))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif command == "train":
        if len(sys.argv) < 3:
            print("Usage: python main.py train <file> [vocab_file] [checkpoint]")
            print("  <file>       - Training data file")
            print("  [vocab_file] - BPE vocabulary file (default: vocab.txt)")
            print("  [checkpoint] - Optional model checkpoint to continue training")
            print("\nBy default, trains a new model from scratch.")
            sys.exit(1)

        train_file = sys.argv[2]
        vocab_file = sys.argv[3] if len(sys.argv) > 3 else "vocab.txt"
        checkpoint_file = sys.argv[4] if len(sys.argv) > 4 else None

        print("=" * 60)
        print("GPT-2 Training")
        print("=" * 60)

        # Check if vocab exists, if not create it
        bpe = BPE(vocab_file)
        if not bpe.load():
            print(f"Vocabulary file '{vocab_file}' not found.")
            print("Creating vocabulary from training data...")

            with open(train_file, 'r') as f:
                text = f.read()

            vocab_size = bpe.get_recommended_vocab_size(text)
            print(f"Recommended vocabulary size: {vocab_size}")
            bpe.set_vocab_size(vocab_size)
            bpe.create_vocab(text)
            bpe.save()
            print(f"Vocabulary saved to {vocab_file}")
        else:
            print(f"Using existing vocabulary from {vocab_file}")

        # Create or load model
        if checkpoint_file:
            print(f"\nLoading model from checkpoint: {checkpoint_file}")
            vocab_size = bpe.get_model_vocab_size()

            # First, try to load the checkpoint to inspect it
            try:
                checkpoint = torch.load(checkpoint_file)

                # Check if vocab size matches
                if 'token_embedding.embedding.weight' in checkpoint:
                    checkpoint_vocab_size = checkpoint['token_embedding.embedding.weight'].shape[0]
                    if checkpoint_vocab_size != vocab_size:
                        print(f"\nWARNING: Vocabulary size mismatch!")
                        print(f"  Checkpoint vocab size: {checkpoint_vocab_size}")
                        print(f"  Current vocab size: {vocab_size}")
                        print(f"\nThis will cause errors during training.")
                        print(f"Please ensure you're using the same vocabulary file that was used to train the checkpoint.")
                        sys.exit(1)

                    # Get embedding dimension from checkpoint
                    embedding_dim = checkpoint['token_embedding.embedding.weight'].shape[1]
                    print(f"Detected model configuration from checkpoint:")
                    print(f"  Vocab size: {checkpoint_vocab_size}")
                    print(f"  Embedding dim: {embedding_dim}")
                else:
                    print("Warning: Could not detect model configuration from checkpoint")
                    embedding_dim = 768  # Default fallback

            except Exception as e:
                print(f"Error inspecting checkpoint: {e}", file=sys.stderr)
                sys.exit(1)

            # Create model with matching configuration
            model = GPT2(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_layers=12,
                num_heads=12,
                max_seq_len=256
            )

            try:
                model.load_state_dict(checkpoint)
                print("Checkpoint loaded successfully!")
                print("Continuing training from checkpoint...")
            except Exception as e:
                print(f"Error loading checkpoint: {e}", file=sys.stderr)
                sys.exit(1)

            trainer = Trainer(
                model=model,
                bpe=bpe,
                max_seq_len=256,
                batch_size=4,
                learning_rate=3e-4
            )
        else:
            print("\nInitializing new GPT-2 Small model from scratch...")
            print("Configuration: 12 layers, 480 dim, 12 heads")

            trainer = Trainer.create_trainer_from_vocab(
                vocab_path=vocab_file,
                embedding_dim=480,
                num_layers=12,
                num_heads=12,
                max_seq_len=256,
                batch_size=4,
                learning_rate=3e-4
            )

        # Train model
        print("\nStarting training...")
        trainer.train(
            filepath=train_file,
            num_epochs=1,
            save_path="gpt2_model.pt"
        )

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Model saved to: gpt2_model.pt")
        print(f"Vocabulary: {vocab_file}")
        print("=" * 60)

    elif command == "predict":
        if len(sys.argv) < 4:
            print("Usage: python main.py predict <prompt> <num_tokens> [model_file] [vocab_file]")
            print("  <prompt>     - Initial text prompt")
            print("  <num_tokens> - Number of tokens to generate")
            print("  [model_file] - Model file (default: gpt2_model.pt)")
            print("  [vocab_file] - Vocabulary file (default: vocab.txt)")
            sys.exit(1)

        prompt = sys.argv[2]
        num_tokens = int(sys.argv[3])
        model_file = sys.argv[4] if len(sys.argv) > 4 else "gpt2_model.pt"
        vocab_file = sys.argv[5] if len(sys.argv) > 5 else "vocab.txt"

        print("=" * 60)
        print("GPT-2 Text Generation")
        print("=" * 60)

        # Load vocabulary
        print(f"Loading vocabulary from {vocab_file}...")
        bpe = BPE(vocab_file)
        if not bpe.load():
            print(f"Error: Vocabulary file '{vocab_file}' not found.", file=sys.stderr)
            sys.exit(1)

        vocab_size = bpe.get_model_vocab_size()
        print(f"Vocabulary: {len(bpe.lookup)} tokens (vocab_size={vocab_size})")

        # Load model - auto-detect configuration from checkpoint
        print(f"Loading model from {model_file}...")

        try:
            # Load checkpoint to detect model configuration
            checkpoint = torch.load(model_file)

            if 'token_embedding.embedding.weight' in checkpoint:
                # Detect configuration from checkpoint
                checkpoint_vocab_size = checkpoint['token_embedding.embedding.weight'].shape[0]
                embedding_dim = checkpoint['token_embedding.embedding.weight'].shape[1]

                if checkpoint_vocab_size != vocab_size:
                    print(f"\nWARNING: Vocabulary size mismatch!", file=sys.stderr)
                    print(f"  Checkpoint vocab size: {checkpoint_vocab_size}", file=sys.stderr)
                    print(f"  Current vocab size: {vocab_size}", file=sys.stderr)
                    print(f"\nPlease use the correct vocabulary file.", file=sys.stderr)
                    sys.exit(1)

                # Detect max_seq_len from position embeddings
                if 'position_embedding.weight' in checkpoint:
                    max_seq_len = checkpoint['position_embedding.weight'].shape[0]
                else:
                    max_seq_len = 1024  # Default fallback

                print(f"Detected model configuration:")
                print(f"  Vocab size: {checkpoint_vocab_size}")
                print(f"  Embedding dim: {embedding_dim}")
                print(f"  Max sequence length: {max_seq_len}")
            else:
                print("Warning: Could not detect model configuration, using defaults")
                embedding_dim = 768
                max_seq_len = 1024

            # Create model with detected configuration
            model = GPT2(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_layers=12,
                num_heads=12,
                max_seq_len=max_seq_len
            )

            # Load state dict
            model.load_state_dict(checkpoint)
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)

        model.eval()

        # Normalize and encode prompt
        print(f"\nPrompt: {prompt}")
        normalized_prompt = bpe.normalize_whitespace(prompt)

        try:
            prompt_ids = bpe.encode(normalized_prompt)
        except ValueError as e:
            print(f"Error encoding prompt: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Prompt tokens: {len(prompt_ids)}")

        # Generate text
        print(f"Generating {num_tokens} tokens...\n")

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        # Generate with temperature sampling
        generated = model.generate(
            prompt_tensor,
            max_new_tokens=num_tokens,
            temperature=0.8,
            top_k=40
        )

        # Decode generated tokens
        generated_ids = generated[0].tolist()
        generated_text = bpe.decode(generated_ids)

        print("Generated text:")
        print("-" * 60)
        print(generated_text)
        print("-" * 60)

        # Also show token-by-token
        print(f"\nTotal tokens generated: {len(generated_ids)}")
        print(f"Original prompt length: {len(prompt_ids)} tokens")
        print(f"New tokens: {len(generated_ids) - len(prompt_ids)} tokens")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: tokenize, encode, decode, train, predict")
        sys.exit(1)

if __name__ == "__main__":
    main()