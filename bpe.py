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
        self.last_id = 1
        self.vocab_size = 1000
        self.min_pair_frequency = min_pair_frequency
        self.whitespace_tokens = set()  # Track which tokens are whitespace

    def get_recommended_vocab_size(self, input):
        words = input.split()
        # find unique words
        unique_words = set(words)
        # A common heuristic is to have a vocabulary size 
        # that is about 2 times the number of unique words in the input
        return len(unique_words) * 2
       
    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size
        
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
                # Track if this is a whitespace token
                if ch.isspace():
                    self.whitespace_tokens.add(token_id)
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

        # Count pairs, but only non-whitespace pairs
        pair_counts = {}
        for i in range(len(input) - 1):
            pair = (input[i], input[i + 1])
            # Skip pairs involving whitespace tokens
            if input[i] in self.whitespace_tokens or input[i + 1] in self.whitespace_tokens:
                continue
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            return input, False

        # Find most common pair (only non-whitespace pairs are in pair_counts)
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

        # Save final vocabulary
        print(f"Saving final vocabulary (size: {len(self.lookup)})...")
        self.save()
        return encoded_input

    def encode(self, input):
        """Encode input text using existing vocabulary with longest-match algorithm.

        Split by whitespace and encode each word using longest matching tokens.
        """
        # Normalize whitespace first
        input = self.normalize_whitespace(input)

        result = []
        i = 0

        while i < len(input):
            # If current character is whitespace, add it directly
            if input[i].isspace():
                if input[i] in self.lookup:
                    result.append(self.lookup[input[i]].id)
                i += 1
                continue

            # Find the end of the current word (next whitespace or end of string)
            word_start = i
            while i < len(input) and not input[i].isspace():
                i += 1
            word = input[word_start:i]

            # Encode the word using longest match
            word_tokens = self._encode_word(word)
            result.extend(word_tokens)

        return result

    def _encode_word(self, word):
        """Encode a single word using longest-match algorithm."""
        if not word:
            return []

        result = []
        pos = 0

        while pos < len(word):
            # Try to find the longest match starting at pos
            longest_match = None
            longest_match_len = 0

            # Try all possible lengths from longest to shortest
            for end in range(len(word), pos, -1):
                substring = word[pos:end]
                if substring in self.lookup:
                    longest_match = substring
                    longest_match_len = end - pos
                    break

            if longest_match:
                # Found a match
                result.append(self.lookup[longest_match].id)
                pos += longest_match_len
            else:
                # No match found - use single character (should always exist from base vocab)
                char = word[pos]
                if char in self.lookup:
                    result.append(self.lookup[char].id)
                else:
                    # Character not in vocabulary - this shouldn't happen if vocab was created properly
                    # but we'll handle it gracefully
                    raise ValueError(f"Character '{char}' not found in vocabulary. Please create vocabulary first.")
                pos += 1

        return result
        
    def save(self):
        with open(self.filename, 'w') as f:
            for token, token_obj in self.lookup.items():
                f.write(f"{token}\t{token_obj.id}\n")
    
    def load(self):
        """Load vocabulary from file if it exists."""
        import os
        if not os.path.exists(self.filename):
            return False

        with open(self.filename, 'r') as f:
            max_id = 0
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                parts = line.split('\t')
                if len(parts) != 2:  # Skip malformed lines
                    continue
                token, id = parts
                token_id = int(id)
                self.lookup[token] = Token(token, token_id)
                self.reverse_lookup[token_id] = token
                # Reconstruct whitespace_tokens set
                if len(token) == 1 and token[0].isspace():
                    self.whitespace_tokens.add(token_id)
                max_id = max(max_id, token_id)

            # Update last_id to continue from where we left off
            self.last_id = max_id + 1

        print(f"Loaded vocabulary from {self.filename}")
        print(f"Vocabulary size: {len(self.lookup)}, Last ID: {self.last_id}")
        return True