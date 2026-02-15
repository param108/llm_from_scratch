# BPE Testing Guide

## Running Tests

This project uses pytest for testing. All tests are located in `test_bpe.py`.

### Installation

Install test dependencies:

```bash
pip install -r requirements.txt
```

### Running All Tests

```bash
pytest
```

Or with verbose output:

```bash
pytest -v
```

### Running Specific Test Suites

Run only encoding tests:
```bash
pytest test_bpe.py::TestBPEEncoding -v
```

Run only vocabulary tests:
```bash
pytest test_bpe.py::TestBPEVocabulary -v
```

Run only optimization tests:
```bash
pytest test_bpe.py::TestBPEOptimization -v
```

### Running Specific Tests

Run a single test:
```bash
pytest test_bpe.py::TestBPEEncoding::test_basic_encoding -v
```

Run parametrized test with specific parameters:
```bash
pytest test_bpe.py::TestBPEEncoding::test_various_inputs[aaa-3-True] -v
```

### Test Coverage

The test suite includes:

- **Encoding Tests** (`TestBPEEncoding`): Basic encoding, repeated patterns, compression ratios
- **Vocabulary Tests** (`TestBPEVocabulary`): Base vocabulary, size limits, merged tokens
- **Persistence Tests** (`TestBPEPersistence`): Save/load functionality
- **Edge Cases** (`TestBPEEdgeCases`): Single character, empty string, unique characters
- **Optimization Tests** (`TestBPEOptimization`): Pair counts reuse, overlapping positions

### Test Output

With the pytest configuration in `pytest.ini`, tests run with:
- Verbose output showing each test
- Short tracebacks for failures
- Automatic test discovery

### Continuous Integration

To run tests in CI/CD:

```bash
pytest --tb=short --disable-warnings
```

### Debug Mode

For detailed output when debugging:

```bash
pytest -v -s --tb=long
```

The `-s` flag shows print statements from the BPE implementation.

## Test Structure

Tests are organized into classes by functionality:

```
test_bpe.py
├── TestBPEEncoding (6 tests)
├── TestBPEVocabulary (3 tests)
├── TestBPEPersistence (1 test)
├── TestBPEEdgeCases (3 tests)
└── TestBPEOptimization (2 tests)
```

## Fixtures

- `temp_vocab_file`: Provides a temporary file path for vocabulary storage
- `bpe_instance`: Provides a fresh BPE instance for each test

These fixtures use pytest's `tmp_path` for automatic cleanup.
