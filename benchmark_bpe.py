"""Benchmark script to measure BPE performance."""

import time
import tempfile
from bpe import BPE


def benchmark_encoding(text, vocab_size, description):
    """Benchmark BPE encoding performance."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        vocab_file = f.name

    bpe = BPE(vocab_file)
    bpe.set_vocab_size(vocab_size)

    # Warm-up run
    _ = bpe.encode(text[:100])

    # Actual benchmark
    bpe = BPE(vocab_file)
    bpe.set_vocab_size(vocab_size)

    start_time = time.time()
    result = bpe.encode(text)
    end_time = time.time()

    elapsed = end_time - start_time
    compression_ratio = len(result) / len(text) if text else 0

    print(f"\n{description}")
    print(f"{'=' * 60}")
    print(f"Input length:        {len(text):,} characters")
    print(f"Encoded length:      {len(result):,} tokens")
    print(f"Vocabulary size:     {len(bpe.lookup):,} tokens")
    print(f"Compression ratio:   {compression_ratio:.2%}")
    print(f"Time:                {elapsed:.4f} seconds")
    print(f"Throughput:          {len(text) / elapsed:,.0f} chars/sec")

    return elapsed, compression_ratio


if __name__ == "__main__":
    print("BPE Performance Benchmark")
    print("=" * 60)

    # Benchmark 1: Small repeated text
    text1 = "hello world " * 100
    benchmark_encoding(text1, 50, "Benchmark 1: Small repeated text")

    # Benchmark 2: Medium text with patterns
    text2 = "the quick brown fox jumps over the lazy dog " * 500
    benchmark_encoding(text2, 100, "Benchmark 2: Medium text with patterns")

    # Benchmark 3: Larger text
    text3 = """
    In computer science, byte pair encoding (BPE) is a data compression technique.
    The algorithm replaces the most frequent pair of bytes in a sequence with a single,
    unused byte. This process repeats until no more replacements can be made.
    """ * 200
    benchmark_encoding(text3, 200, "Benchmark 3: Larger text")

    # Benchmark 4: Very large text
    text4 = "abcdefghijklmnopqrstuvwxyz " * 1000
    benchmark_encoding(text4, 300, "Benchmark 4: Very large text")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
