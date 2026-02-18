# bpe.py

class Token:
    def __init__(self, token, id):
        self.token = token
        self.id = id

class BPE:
    def __init__(self, filename, min_pair_frequency=5):
        self.lookup = {}
        self.reverse_lookup = {}
        self.filename = filename
        self.last_id = 0
        self.vocab_size = 1000
        self.min_pair_frequency = min_pair_frequency
        self.eot_token_id = None  # Special End-of-Text token ID

    def get_recommended_vocab_size(self, input):
        words = input.split()
        # find unique words
        unique_words = set(words)
        # A common heuristic is to have a vocabulary size
        # that is about 2 times the number of unique words in the input
        return len(unique_words) * 2

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size

    def get_model_vocab_size(self):
        """
        Get the vocabulary size to use when creating a model.

        This returns the maximum token ID that can be generated + 1,
        which is needed for PyTorch embedding layers that expect
        token IDs in range [0, vocab_size).

        Returns:
            The vocabulary size for model creation (last_id)
        """
        if len(self.lookup) == 0:
            return 0
        # The vocab size for the model should be last_id because
        # token IDs can range from 0 to last_id-1
        return self.last_id

    def get_eot_token_id(self):
        """Get the End-of-Text token ID.

        Returns:
            The EOT token ID, or None if not set
        """
        return self.eot_token_id

    def get_next_id(self):
        id = self.last_id
        self.last_id += 1
        return id

    def normalize_whitespace(self, input):
        """Normalize whitespace characters in the input.
        - Replace all whitespace except CRLF with single space
        - CRLF/CR/LF preceded or followed by whitespace becomes single CRLF
        """
        import re
        # First, normalize line breaks with surrounding whitespace to single CRLF
        # Match: optional whitespace + line break + optional whitespace
        input = re.sub(r'[ \t]*(?:\r\n|\r|\n)[ \t]*', '\n', input)
        # Replace all other whitespace (tabs, multiple spaces, etc.) with single space
        input = re.sub(r'[ \t]+', ' ', input)
        return input

    def base_vocab(self, input):
        # Normalize whitespace first
        input = self.normalize_whitespace(input)

        encoded_input = []
        had_existing_vocab = len(self.lookup) > 0
        new_chars = 0

        # generate base vocabulary (or extend existing one)
        for ch in input:
            if ch not in self.lookup:
                new_chars += 1
                token_id = self.get_next_id()
                self.lookup[str(ch)] = Token(str(ch), token_id)
                self.reverse_lookup[token_id] = str(ch)
            encoded_input.append(self.lookup[str(ch)].id)

        if not had_existing_vocab:
            print("Base vocabulary:")
            print("number of tokens:", len(self.lookup))
        else:
            print(f"Extended vocabulary (added {new_chars} new chars)")
            print(f"Total vocabulary size: {len(self.lookup)}")
        print("input length:", len(encoded_input))
        return encoded_input

    def bpe_merge(self, input):
        """Simple BPE merge - count pairs, find most common, merge."""
        if len(input) < 2:
            return input, False

        # Count pairs, but skip pairs where left token ends with space
        pair_counts = {}
        for i in range(len(input) - 1):
            pair = (input[i], input[i + 1])
            # Skip if the left token ends with a space
            left_token_str = self.reverse_lookup[input[i]]
            if left_token_str.endswith(' '):
                continue
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            return input, False

        # Find most common pair
        most_common_pair = max(pair_counts, key=pair_counts.get)
        max_count = pair_counts[most_common_pair]

        # If the most common pair appears less than min_pair_frequency, stop encoding
        if max_count < self.min_pair_frequency:
            return input, False

        # Create new token for the merge
        new_token_id = self.get_next_id()
        new_token_str = self.reverse_lookup[most_common_pair[0]] + self.reverse_lookup[most_common_pair[1]]
        self.lookup[new_token_str] = Token(new_token_str, new_token_id)
        self.reverse_lookup[new_token_id] = new_token_str

        # Replace all occurrences of the most common pair
        new_input = []
        i = 0
        while i < len(input):
            if i < len(input) - 1 and (input[i], input[i + 1]) == most_common_pair:
                new_input.append(new_token_id)
                i += 2
            else:
                new_input.append(input[i])
                i += 1

        return new_input, True

    def create_vocab(self, input):
        """Create vocabulary from input text using BPE algorithm."""
        # Try to load existing vocabulary first (only if we don't have one already)
        if len(self.lookup) == 0:
            loaded = self.load()
            if loaded:
                print(f"Continuing from loaded vocabulary (size: {len(self.lookup)})")

        encoded_input = self.base_vocab(input)
        last_checkpoint = (len(self.lookup) // 1000) * 1000  # Round down to nearest 1000

        while len(self.lookup) < self.vocab_size:
            prev_vocab_size = len(self.lookup)
            encoded_input, should_continue = self.bpe_merge(encoded_input)
            print("Required vocabulary size:", self.vocab_size, "Current vocabulary size:", len(self.lookup), "Encoded input length:", len(encoded_input))

            # Save checkpoint every 100 tokens
            current_checkpoint = (len(self.lookup) // 100) * 100
            if current_checkpoint > last_checkpoint and current_checkpoint > 0:
                print(f"Saving checkpoint at vocabulary size {len(self.lookup)}...")
                self.save()
                last_checkpoint = current_checkpoint

            # Break if no more merges are possible or max pair count < min_pair_frequency
            if not should_continue or len(self.lookup) == prev_vocab_size or len(encoded_input) < 2:
                break

        # Add special <EOT> token with the max vocab ID
        self._add_eot_token()

        # Save final vocabulary
        print(f"Saving final vocabulary (size: {len(self.lookup)})...")
        self.save()
        return encoded_input

    def _add_eot_token(self):
        """Add the special <EOT> (End-of-Text) token with max vocab ID."""
        eot_str = "<EOT>"
        if eot_str in self.lookup:
            # EOT already exists in vocabulary (from training data)
            # Just mark it as the special EOT token
            self.eot_token_id = self.lookup[eot_str].id
            print(f"Marked existing <EOT> token (ID: {self.eot_token_id}) as special EOT token")
        else:
            # EOT doesn't exist, create it as a new token
            self.eot_token_id = self.get_next_id()
            self.lookup[eot_str] = Token(eot_str, self.eot_token_id)
            self.reverse_lookup[self.eot_token_id] = eot_str
            print(f"Added special <EOT> token with ID: {self.eot_token_id}")

    def encode(self, input):
        """Encode input text using existing vocabulary with greedy forward matching."""
        # Normalize whitespace first
        input = self.normalize_whitespace(input)

        result = []
        current = ""
        last_match = None
        last_match_end = 0
        pos = 0

        while pos < len(input):
            current += input[pos]

            # Check for <EOT> token
            if current == "<EOT>" and "<EOT>" in self.lookup:
                result.append(self.lookup["<EOT>"].id)
                current = ""
                last_match = None
                pos += 1
                continue

            if current in self.lookup:
                # Current string exists in vocab, remember it and keep growing
                last_match = current
                last_match_end = pos + 1
                pos += 1
            else:
                # Current string not in vocab
                if last_match is not None:
                    # Emit the previous match
                    result.append(self.lookup[last_match].id)
                    # Restart from where that match ended
                    pos = last_match_end
                    current = ""
                    last_match = None
                else:
                    # Single character not in vocab
                    raise ValueError(f"Character '{input[pos]}' not found in vocabulary. Please create vocabulary first.")

        # Don't forget the last match
        if last_match is not None:
            result.append(self.lookup[last_match].id)
        elif current:
            raise ValueError(f"Token '{current}' not found in vocabulary. Please create vocabulary first.")

        return result

    def decode(self, token_ids):
        """Decode a list of token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        result = []
        for token_id in token_ids:
            if token_id in self.reverse_lookup:
                result.append(self.reverse_lookup[token_id])
            else:
                raise ValueError(f"Token ID {token_id} not found in vocabulary.")
        return ''.join(result)

    def _escape_token(self, s):
        """Escape special characters for saving to file."""
        return s.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

    def _unescape_token(self, s):
        """Unescape special characters when loading from file."""
        result = []
        i = 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s):
                c = s[i + 1]
                if c == 'n':
                    result.append('\n')
                elif c == 'r':
                    result.append('\r')
                elif c == 't':
                    result.append('\t')
                elif c == '\\':
                    result.append('\\')
                else:
                    result.append(s[i])
                    result.append(c)
                i += 2
            else:
                result.append(s[i])
                i += 1
        return ''.join(result)

    def save(self):
        with open(self.filename, 'w') as f:
            for token, token_obj in self.lookup.items():
                escaped = self._escape_token(token)
                f.write(f"{escaped}\t{token_obj.id}\n")

    def load(self):
        """Load vocabulary from file if it exists."""
        import os
        if not os.path.exists(self.filename):
            return False

        with open(self.filename, 'r') as f:
            max_id = 0
            for line in f:
                line = line.rstrip('\r\n')
                if not line:  # Skip empty lines
                    continue
                parts = line.split('\t')
                if len(parts) != 2:  # Skip malformed lines
                    continue
                raw_token, id = parts
                token = self._unescape_token(raw_token)
                token_id = int(id)
                self.lookup[token] = Token(token, token_id)
                self.reverse_lookup[token_id] = token
                # Check for <EOT> token
                if token == "<EOT>":
                    self.eot_token_id = token_id
                max_id = max(max_id, token_id)

            # Update last_id to continue from where we left off
            # Never decrease below initial value to protect reserved IDs
            self.last_id = max(self.last_id, max_id + 1)

        print(f"Loaded vocabulary from {self.filename}")
        print(f"Vocabulary size: {len(self.lookup)}, Last ID: {self.last_id}")
        return True
