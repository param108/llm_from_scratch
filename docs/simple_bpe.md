# Simple BPE Algorithm with Early Exit

## Overview

This document describes the current BPE (Byte Pair Encoding) implementation - a straightforward, easy-to-understand algorithm with configurable early exit based on pair frequency.

## Algorithm Design

The implementation prioritizes simplicity and readability over complex optimizations. Each iteration:
1. Counts all pairs in the input
2. Finds the most common pair
3. Checks if it meets the frequency threshold
4. Merges all occurrences of that pair

## Key Features

### 1. Simple Pair Counting

```python
def bpe_merge(self, input):
    """Simple BPE merge - count pairs, find most common, merge."""
    if len(input) < 2:
        return input, False

    # Count all pairs
    pair_counts = {}
    for i in range(len(input) - 1):
        pair = (input[i], input[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
```

**Why simple counting?**
- Easy to understand and debug
- No complex state management
- Clean separation between iterations
- Sufficient performance for most use cases

### 2. Early Exit on Low Frequency

```python
# Find most common pair
most_common_pair = max(pair_counts, key=pair_counts.get)
max_count = pair_counts[most_common_pair]

# If the most common pair appears less than min_pair_frequency, stop encoding
if max_count < self.min_pair_frequency:
    return input, False
```

**Configurable Threshold:**
```python
# Default threshold of 5
bpe = BPE("vocab.txt", min_pair_frequency=5)

# Or use custom threshold
bpe = BPE("vocab.txt", min_pair_frequency=10)

# For maximum compression (no early exit)
bpe = BPE("vocab.txt", min_pair_frequency=1)
```

**Why early exit?**
- Prevents over-fitting to rare patterns
- Avoids creating tokens for infrequent pairs
- Reduces vocabulary size for sparse data
- Improves generalization

### 3. Clean Merging Logic

```python
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
```

**Key points:**
- Handles overlapping pairs correctly
- Single pass through input
- Clear token creation and registration

## Complete Encoding Process

```python
def encode(self, input):
    encoded_input = self.base_vocab(input)
    while len(self.lookup) < self.vocab_size:
        prev_vocab_size = len(self.lookup)
        encoded_input, should_continue = self.bpe_merge(encoded_input)
        print("Required vocabulary size:", self.vocab_size,
              "Current vocabulary size:", len(self.lookup),
              "Encoded input length:", len(encoded_input))

        # Break if no more merges are possible or max pair count < min_pair_frequency
        if not should_continue or len(self.lookup) == prev_vocab_size or len(encoded_input) < 2:
            break
    return encoded_input
```

**Stopping conditions:**
1. Reached target vocabulary size
2. Most common pair frequency < threshold
3. No vocabulary growth (all pairs unique)
4. Input too short to have pairs

## Usage Examples

### Basic Usage

```python
from embeddings import BPE

# Create BPE instance with default threshold
bpe = BPE("vocab.txt")

# Encode text
text = "hello world hello world"
bpe.set_vocab_size(50)
encoded = bpe.encode(text)

# Save vocabulary
bpe.save()
```

### Custom Frequency Threshold

```python
# High threshold - more aggressive early exit
bpe = BPE("vocab.txt", min_pair_frequency=10)
encoded = bpe.encode(text)
# Will stop when most common pair appears < 10 times

# Low threshold - more compression
bpe = BPE("vocab.txt", min_pair_frequency=2)
encoded = bpe.encode(text)
# Will continue until pairs appear < 2 times
```

### Comparing Thresholds

```python
import pytest

def test_threshold_comparison(temp_vocab_file):
    text = "hello world " * 10

    # Low threshold - more compression
    bpe_low = BPE(temp_vocab_file, min_pair_frequency=1)
    bpe_low.set_vocab_size(100)
    encoded_low = bpe_low.encode(text)

    # High threshold - less compression
    bpe_high = BPE(temp_vocab_file, min_pair_frequency=10)
    bpe_high.set_vocab_size(100)
    encoded_high = bpe_high.encode(text)

    # Lower threshold should achieve better compression
    assert len(encoded_low) <= len(encoded_high)
    assert len(bpe_low.lookup) >= len(bpe_high.lookup)
```

## Time Complexity

### Per Iteration
- **Count pairs**: O(n) where n is input length
- **Find max**: O(p) where p is number of unique pairs
- **Merge pairs**: O(n)
- **Total per iteration**: O(n + p)

### Full Encoding
- **Iterations**: O(v) where v is target vocabulary size
- **Total**: O(v × n) in worst case
- **Typical**: O(v × n) but n decreases each iteration as pairs merge

### Space Complexity
- **Input array**: O(n)
- **Pair counts**: O(p) where p ≤ n
- **Vocabulary**: O(v)
- **Total**: O(n + v)

## Performance Characteristics

### Strengths
✅ **Simple and readable** - Easy to understand and modify
✅ **Correct** - Handles all edge cases properly
✅ **Configurable** - Threshold parameter for different use cases
✅ **Maintainable** - No complex state management
✅ **Predictable** - Consistent behavior across inputs

### Trade-offs
⚠️ **Recounts pairs** - Doesn't maintain state between iterations
⚠️ **Sequential** - Single-threaded execution
⚠️ **Memory copies** - Creates new arrays each iteration

### When to Use
**Best for:**
- Small to medium inputs (<100K characters)
- Readability and maintainability priorities
- Educational purposes
- Prototyping and experimentation
- Single-core systems

**Consider alternatives for:**
- Very large inputs (>1M characters)
- Production systems requiring maximum throughput
- Batch processing of many documents

## Design Philosophy

This implementation follows these principles:

1. **Simplicity over optimization** - Prefer readable code to complex optimizations
2. **Correctness first** - Ensure algorithm works correctly before optimizing
3. **Configurability** - Make key parameters adjustable
4. **Clean separation** - Each iteration is independent
5. **No premature optimization** - Start simple, optimize if needed

## Testing

The implementation has comprehensive test coverage:

```python
# 20 tests total
class TestBPEBasic:
    # 5 basic functionality tests

class TestBPEEncoding:
    # 6 encoding tests

class TestBPEVocabulary:
    # 4 vocabulary tests

class TestBPEEarlyExit:
    # 5 early exit threshold tests
```

All tests use `min_pair_frequency=1` by default to test maximum compression, with specific tests for threshold behavior.

## Comparison with Other Approaches

| Aspect | Simple BPE | Optimized Incremental | Parallel D&C |
|--------|-----------|---------------------|--------------|
| **Complexity** | Low | High | Medium |
| **Speed** | Good | Better | Best |
| **Memory** | O(n) | O(n + p×k) | O(n×threads) |
| **Readability** | Excellent | Poor | Medium |
| **Maintenance** | Easy | Hard | Medium |
| **Best for** | General use | Small inputs | Large inputs |

## Future Enhancements

Possible improvements while maintaining simplicity:

1. **Adaptive threshold** - Automatically adjust min_pair_frequency based on input
2. **Decoding support** - Add method to decode token sequences back to text
3. **Vocabulary reuse** - Load and extend existing vocabularies
4. **Statistics** - Track merge history and compression ratios
5. **Input validation** - Better error handling for edge cases

## References

- Implementation: [bpe.py](../bpe.py)
- Tests: [test_bpe.py](../test_bpe.py)
- Main documentation: [README.md](../README.md)
- Historical optimizations: [bpe_optimizations.md](bpe_optimizations.md) (reference only)
- Historical parallel: [parallel_bpe.md](parallel_bpe.md) (reference only)

## Conclusion

This simple BPE implementation provides:

✅ **Clear, understandable code** - Easy to read and modify
✅ **Configurable early exit** - Prevents over-fitting via min_pair_frequency
✅ **Correct behavior** - All 20 tests pass
✅ **Good performance** - Sufficient for most use cases
✅ **Low maintenance** - No complex state to manage

For most applications, this straightforward approach is the right choice. Optimize only when profiling shows it's necessary.
