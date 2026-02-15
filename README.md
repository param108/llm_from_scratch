# BPE (Byte Pair Encoding) Implementation

A simple, readable Python implementation of the Byte Pair Encoding algorithm for text tokenization.

## Features

- ✅ **Simple and readable** - Easy to understand and modify
- ✅ **Configurable early exit** - Prevents over-fitting with frequency threshold
- ✅ **Correct behavior** - Handles all edge cases properly
- ✅ **Comprehensive test suite** - 20 pytest tests with full coverage
- ✅ **Well-documented** - Clear algorithm explanation and examples

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Tokenize a file
python main.py tokenize input.txt
```

### Python API

```python
from bpe import BPE

# Create BPE instance with default threshold (5)
bpe = BPE("vocab.txt")

# Or specify custom frequency threshold
bpe = BPE("vocab.txt", min_pair_frequency=10)

# Encode text
text = "hello world hello world"
bpe.set_vocab_size(50)
encoded = bpe.encode(text)

# Save vocabulary
bpe.save()

# Load vocabulary
bpe.load()
```

## Documentation

- **[Simple BPE Algorithm](docs/simple_bpe.md)** - Current implementation with early exit feature
- **[Testing Guide](README_TESTS.md)** - How to run and write tests
- **[BPE Optimizations Guide](docs/bpe_optimizations.md)** - Historical incremental optimization (reference only)
- **[Parallel BPE Algorithm](docs/parallel_bpe.md)** - Historical parallel implementation (reference only)

## Project Structure

```
.
├── bpe.py                    # Main BPE implementation (simple with early exit)
├── main.py                   # CLI interface
├── test_bpe.py              # Pytest test suite (20 tests)
├── benchmark_bpe.py         # Performance benchmarking
├── docs/
│   ├── simple_bpe.md        # Current algorithm documentation
│   ├── bpe_optimizations.md # Historical optimizations (reference)
│   └── parallel_bpe.md      # Historical parallel version (reference)
├── pytest.ini               # Pytest configuration
└── requirements.txt         # Python dependencies
```

## Algorithm Details

### Early Exit Feature

The implementation includes a configurable `min_pair_frequency` parameter (default: 5) that stops encoding when the most common pair appears fewer times than the threshold. This:

- Prevents over-fitting to rare patterns
- Reduces vocabulary size for sparse data
- Improves generalization
- Speeds up encoding by avoiding unnecessary merges

### Run Benchmarks

```bash
python benchmark_bpe.py
```

## Testing

Run the test suite:

```bash
# All tests
pytest

# Verbose output
pytest -v

# Specific test class
pytest test_bpe.py::TestBPEEncoding -v
```

See [README_TESTS.md](README_TESTS.md) for detailed testing documentation.

## Key Features

1. **Simple Algorithm**: Count pairs, find max, merge - easy to understand
2. **Early Exit**: Configurable frequency threshold prevents over-fitting
3. **Clean Code**: No complex state management between iterations
4. **Well-Tested**: 20 comprehensive tests covering all functionality
5. **Flexible**: Adjustable vocabulary size and frequency threshold

See [docs/simple_bpe.md](docs/simple_bpe.md) for detailed explanation.

## Algorithm Complexity

| Operation | Time Complexity |
|-----------|----------------|
| Count pairs | O(n) |
| Find max pair | O(unique_pairs) |
| Merge pairs | O(n) |
| **Per iteration** | **O(n + unique_pairs)** |
| **Full encoding** | **O(v × n)** where v = vocab size |

Simple and predictable performance characteristics.

## Requirements

- Python 3.8+
- pytest (for testing)

## Contributing

Contributions welcome! Areas for improvement:
- Additional test cases for edge cases
- Decoding functionality
- Support for pre-trained vocabularies
- Adaptive threshold selection
- Performance optimizations (if needed)

## License

MIT License - see LICENSE file for details

## References

- Original BPE Paper: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- Algorithm documentation: [docs/simple_bpe.md](docs/simple_bpe.md)
- Historical implementations:
  - [docs/parallel_bpe.md](docs/parallel_bpe.md) - Parallel version (reference)
  - [docs/bpe_optimizations.md](docs/bpe_optimizations.md) - Incremental optimization (reference)

## Implementation History

- Simple BPE algorithm with early exit feature
- Comprehensive test suite with 20 test cases
- Historical optimized and parallel versions (see docs/)
- Detailed documentation and examples
