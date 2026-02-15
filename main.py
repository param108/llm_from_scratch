import sys
import json
from bpe import BPE

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [args...]")
        print("Commands:")
        print("  tokenize <file>              - Create vocabulary from file")
        print("  encode <text> [vocab_file]   - Encode text using vocabulary")
        print("  decode <json_array> [vocab_file] - Decode token IDs to text")
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

    else:
        print(f"Unknown command: {command}")
        print("Available commands: tokenize, encode, decode")
        sys.exit(1)

if __name__ == "__main__":
    main()